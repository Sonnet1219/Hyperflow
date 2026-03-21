import argparse
import json
from sentence_transformers import SentenceTransformer
from hyperflow.config import HyperflowConfig
from hyperflow.engine import Hyperflow
import os
import warnings
from hyperflow.evaluate import Evaluator
from hyperflow.utils import LLM_Model
from hyperflow.utils import setup_logging
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="novel", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--passage_ratio", type=float, default=2, help="The ratio for passage")
    parser.add_argument("--diffusion_alpha", type=float, default=0.85, help="Diffusion weight (higher=more exploration)")
    parser.add_argument("--diffusion_max_iter", type=int, default=10, help="Max iterations for diffusion convergence")
    parser.add_argument("--convergence_tol", type=float, default=1e-4, help="L2 norm convergence threshold for diffusion")
    parser.add_argument("--sentence_gate_threshold", type=float, default=0.5, help="Block sentences with query-similarity below this threshold")
    parser.add_argument("--diffusion_top_k", type=int, default=10, help="Activate top-K entities per diffusion round")
    parser.add_argument("--reranker_model", type=str, default="Qwen/Qwen3-Reranker-4B", help="Local or Hub path for the reranker model")
    parser.add_argument("--reranker_candidate_top_k", type=int, default=30, help="How many retrieval candidates to pass into the reranker")
    parser.add_argument("--reranker_batch_size", type=int, default=16, help="Batch size for reranker scoring")
    parser.add_argument("--reranker_max_length", type=int, default=4096, help="Max token length for reranker inputs")
    parser.add_argument("--question_limit", type=int, default=None, help="Optional limit for the number of questions to run")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip the answer evaluation stage")
    return parser.parse_args()


def load_dataset(dataset_name):
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages

def load_embedding_model(embedding_model):
    embedding_model = SentenceTransformer(embedding_model,device="cuda")
    return embedding_model

def main():
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()
    output_dir = f"results/{args.dataset_name}/{time_str}"
    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.dataset_name)
    if args.question_limit is not None:
        questions = questions[:args.question_limit]
    setup_logging(f"{output_dir}/log.txt")
    llm_model = LLM_Model(args.llm_model)
    config = HyperflowConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        passage_ratio=args.passage_ratio,
        diffusion_alpha=args.diffusion_alpha,
        diffusion_max_iter=args.diffusion_max_iter,
        convergence_tol=args.convergence_tol,
        sentence_gate_threshold=args.sentence_gate_threshold,
        diffusion_top_k=args.diffusion_top_k,
        reranker_model_name=args.reranker_model,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length
    )
    model = Hyperflow(global_config=config)
    model.index(passages)
    questions = model.qa(questions)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    if not args.skip_evaluation:
        evaluator = Evaluator(llm_model=llm_model, predictions_path=f"{output_dir}/predictions.json")
        evaluator.evaluate(max_workers=args.max_workers)
if __name__ == "__main__":
    main()
