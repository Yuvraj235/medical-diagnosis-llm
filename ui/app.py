"""
ui/app.py
Gradio-based UI for Medical RAG Pipeline.
Features:
  - Medical Q&A with evidence display
  - Real-time safety indicators
  - Evidence explainability visualization
  - Evaluation dashboard
  - Batch evaluation runner
"""
import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, MEDICAL_DISCLAIMER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Global pipeline (lazy init)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline.rag_pipeline import MedicalRAGPipeline
        _pipeline = MedicalRAGPipeline()
        _pipeline.initialize(lazy=True)
    return _pipeline


# ─────────────────────────────────────────────
# QUERY HANDLER
# ─────────────────────────────────────────────

def answer_medical_question(
    question: str,
    top_k: int,
    show_evidence: bool,
    show_explainability: bool,
):
    """Main query handler called by Gradio."""
    if not question.strip():
        return (
            "❌ Please enter a medical question.",
            "", "", "", "N/A", "N/A", "N/A"
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.query(
            question,
            top_k=int(top_k),
            return_evidence=show_evidence,
            return_explainability=show_explainability,
        )

        # ── Answer ──────────────────────────────────────
        answer_text = result.get("answer", "No answer generated.")

        # Emergency banner
        if result.get("emergency"):
            answer_text = f"🚨 {result.get('emergency_message', '')}\n\n{answer_text}"

        # If unsafe
        if not result.get("safe", True):
            warnings = "; ".join(result.get("safety_warnings", []))
            answer_text = f"⚠️ Safety Warning: {warnings}\n\n{answer_text}"

        # ── Predicted Label ──────────────────────────────
        label = result.get("predicted_label", "N/A").upper()
        label_display = {
            "YES": "✅ YES",
            "NO": "❌ NO",
            "MAYBE": "🤔 MAYBE",
        }.get(label, f"❓ {label}")

        # ── Evidence HTML ────────────────────────────────
        evidence_html = ""
        if show_evidence and result.get("retrieved_chunks"):
            chunks = result["retrieved_chunks"]
            evidence_html = f"<h4>📚 Retrieved Evidence ({len(chunks)} chunks)</h4>"
            for c in chunks:
                score = c.get("score", 0.0)
                color = "#d4edda" if score > 0.7 else "#fff3cd" if score > 0.5 else "#f8f9fa"
                pid = c.get("metadata", {}).get("pubmed_id", "N/A")
                evidence_html += f"""
                <div style="border:1px solid #ddd; border-radius:8px; padding:10px; 
                            margin:6px 0; background:{color};">
                    <b>Rank #{c.get('rank', '?')}</b> — PubMed ID: {pid} — 
                    Score: <b>{score:.3f}</b>
                    <p style="margin:6px 0; font-size:0.9em;">{c['chunk'][:300]}...</p>
                </div>"""

        # ── Explainability HTML ──────────────────────────
        explain_html = ""
        if show_explainability and result.get("evidence_html"):
            confidence = result.get("explainability", {}).get("confidence", 0)
            explain_html = f"""
            <h4>🔍 Evidence Attribution (Confidence: {confidence:.1%})</h4>
            {result['evidence_html']}
            """
            if result.get("explainability", {}).get("attribution_summary"):
                explain_html += f"""
                <pre style="background:#f0f0f0; padding:10px; border-radius:6px; font-size:0.85em;">
{result['explainability']['attribution_summary']}</pre>"""

        # ── Stats ────────────────────────────────────────
        latency = f"{result.get('latency_ms', 0):.0f} ms"
        retrieval_score = f"{result.get('avg_retrieval_score', 0):.3f}"
        safety_score = f"{result.get('safety_scores', {}).get('toxicity', 0):.3f}"

        return (
            answer_text,
            label_display,
            evidence_html,
            explain_html,
            latency,
            retrieval_score,
            safety_score,
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return (
            f"❌ Error: {str(e)}\n\nPlease ensure the pipeline is initialized. "
            "Run: python data/download_data.py && python embeddings/pubmedbert_embedder.py",
            "N/A", "", "", "N/A", "N/A", "N/A"
        )


# ─────────────────────────────────────────────
# EVALUATION HANDLER
# ─────────────────────────────────────────────

def run_evaluation_ui(n_samples: int, use_mock: bool):
    """Run evaluation and return results for display."""
    try:
        from evaluation.run_evaluation import run_evaluation
        from dataclasses import asdict

        result = run_evaluation(
            n_samples=int(n_samples),
            save_plots=True,
            use_mock=use_mock,
        )

        metrics = asdict(result)

        # Build results table
        table_data = [
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            ["F1 Macro", f"{metrics['f1_macro']:.4f}"],
            ["F1 Weighted", f"{metrics['f1_weighted']:.4f}"],
            ["BLEU-1", f"{metrics['bleu_1']:.4f}"],
            ["BLEU-4", f"{metrics['bleu_4']:.4f}"],
            ["ROUGE-1 F", f"{metrics['rouge_1_f']:.4f}"],
            ["ROUGE-2 F", f"{metrics['rouge_2_f']:.4f}"],
            ["ROUGE-L F", f"{metrics['rouge_l_f']:.4f}"],
            ["BERTScore F1", f"{metrics['bertscore_f1']:.4f}"],
            ["Fluency", f"{metrics['fluency_score']:.4f}"],
            ["Relevance", f"{metrics['relevance_score']:.4f}"],
            ["Coherence", f"{metrics['coherence_score']:.4f}"],
            ["Faithfulness", f"{metrics['faithfulness_score']:.4f}"],
            ["Avg Toxicity", f"{metrics['avg_toxicity']:.4f}"],
            ["Toxicity Rate", f"{metrics['toxicity_rate']:.4f}"],
            ["Hit Rate @1", f"{metrics['hit_rate_at_1']:.4f}"],
            ["Hit Rate @5", f"{metrics['hit_rate_at_5']:.4f}"],
            ["MRR", f"{metrics['mrr']:.4f}"],
            ["Avg Latency (ms)", f"{metrics['avg_latency_ms']:.1f}"],
        ]

        # Try to load and display the overview plot
        plots_dir_base = RESULTS_DIR
        plot_files = list(Path(plots_dir_base).glob("plots_*/evaluation_overview.png"))
        plot_path = str(max(plot_files, key=os.path.getctime)) if plot_files else None

        return table_data, plot_path, "✅ Evaluation complete!"

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return [], None, f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────

def build_ui():
    import gradio as gr

    # Sample questions for quick testing
    sample_questions = [
        "Does metformin reduce cardiovascular mortality in type 2 diabetes patients?",
        "Is cognitive behavioral therapy effective for treating major depressive disorder?",
        "Do statins reduce the risk of stroke in patients with high cholesterol?",
        "Can aspirin therapy prevent recurrent myocardial infarction?",
        "Is remdesivir effective in reducing mortality from COVID-19?",
    ]

    with gr.Blocks(title="🏥 Medical RAG — Biomedical QA System") as demo:

        # ── Header ──────────────────────────────────────────────
        gr.HTML("""
        <div class="header">
            <h1>🏥 Medical RAG Pipeline</h1>
            <p>Biomedical Question Answering with Retrieval-Augmented Generation</p>
            <p style="font-size:0.85em; opacity:0.8;">
                PubMedBERT Embeddings + FAISS Retrieval + LoRA Fine-tuned LLM + Clinical Guardrails
            </p>
        </div>
        """)

        # ── Tabs ─────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Medical Q&A ─────────────────────────────
            with gr.Tab("💬 Medical Q&A", id="qa"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Ask a Medical Question")
                        question_input = gr.Textbox(
                            label="Medical Question",
                            placeholder="E.g., Does aspirin reduce cardiovascular risk in elderly patients?",
                            lines=3,
                            max_lines=5,
                        )
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="Evidence Chunks (Top-K)"
                            )
                        with gr.Row():
                            show_evidence = gr.Checkbox(value=True, label="Show Retrieved Evidence")
                            show_explain = gr.Checkbox(value=True, label="Show Explainability")

                        with gr.Row():
                            submit_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                        gr.Markdown("#### 💡 Sample Questions")
                        for i, q in enumerate(sample_questions):
                            gr.Button(q[:70] + "...", size="sm").click(
                                fn=lambda x=q: x,
                                outputs=question_input
                            )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 Results")

                        with gr.Row():
                            label_output = gr.Textbox(
                                label="Predicted Answer",
                                interactive=False,
                                scale=1
                            )
                            with gr.Column(scale=3):
                                with gr.Row():
                                    latency_display = gr.Textbox(label="⏱️ Latency", interactive=False, scale=1)
                                    retrieval_display = gr.Textbox(label="📊 Avg Retrieval Score", interactive=False, scale=1)
                                    safety_display = gr.Textbox(label="🛡️ Toxicity Score", interactive=False, scale=1)

                        answer_output = gr.Textbox(
                            label="📝 Generated Answer",
                            lines=8,
                            interactive=False,
                        )

                        gr.HTML("<hr>")
                        evidence_output = gr.HTML(label="📚 Retrieved Evidence")
                        explain_output = gr.HTML(label="🔍 Evidence Explainability")

                # Handlers
                submit_btn.click(
                    fn=answer_medical_question,
                    inputs=[question_input, top_k_slider, show_evidence, show_explain],
                    outputs=[answer_output, label_output, evidence_output, explain_output,
                             latency_display, retrieval_display, safety_display],
                )
                clear_btn.click(
                    fn=lambda: ("", "", "", "", "", "N/A", "N/A", "N/A"),
                    outputs=[question_input, answer_output, label_output, evidence_output,
                             explain_output, latency_display, retrieval_display, safety_display],
                )
                question_input.submit(
                    fn=answer_medical_question,
                    inputs=[question_input, top_k_slider, show_evidence, show_explain],
                    outputs=[answer_output, label_output, evidence_output, explain_output,
                             latency_display, retrieval_display, safety_display],
                )

            # ── Tab 2: Evaluation Dashboard ────────────────────
            with gr.Tab("📊 Evaluation", id="eval"):
                gr.Markdown("""
                ### 📊 Quantitative Evaluation Dashboard
                Run comprehensive evaluation on the PubMedQA test set.
                """)

                with gr.Row():
                    eval_n_samples = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10,
                        label="Number of Test Samples"
                    )
                    use_mock = gr.Checkbox(
                        value=False,
                        label="Use Mock Results (faster, no model loading)"
                    )
                    eval_btn = gr.Button("🚀 Run Evaluation", variant="primary")

                eval_status = gr.Textbox(label="Status", interactive=False)
                eval_plot = gr.Image(label="📈 Evaluation Overview", show_label=True)

                eval_table = gr.Dataframe(
                    headers=["Metric", "Score"],
                    label="📋 Quantitative Results",
                    wrap=True,
                )

                eval_btn.click(
                    fn=run_evaluation_ui,
                    inputs=[eval_n_samples, use_mock],
                    outputs=[eval_table, eval_plot, eval_status],
                )

            # ── Tab 3: System Info ──────────────────────────────
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown(f"""
                ## 🏗️ Architecture

                ```
                User Query
                    ↓
                ┌─────────────────────────────────────────┐
                │         Input Safety Check              │
                │  (Blocked topics + Toxicity detection)  │
                └──────────────────┬──────────────────────┘
                                   ↓
                ┌─────────────────────────────────────────┐
                │     PubMedBERT Query Encoding           │
                │  microsoft/BiomedNLP-BiomedBERT-base    │
                └──────────────────┬──────────────────────┘
                                   ↓
                ┌─────────────────────────────────────────┐
                │     FAISS Semantic Retrieval            │
                │   Top-K=5 chunks from PubMedQA corpus  │
                └──────────────────┬──────────────────────┘
                                   ↓
                ┌─────────────────────────────────────────┐
                │  LoRA Fine-tuned LLM (BioGPT/BioMistral)│
                │   Rank=16 | Alpha=32 | PubMedQA-L data  │
                └──────────────────┬──────────────────────┘
                                   ↓
                ┌─────────────────────────────────────────┐
                │       Output Safety Check               │
                │  (Toxicity + Bias + Confidence)         │
                └──────────────────┬──────────────────────┘
                                   ↓
                ┌─────────────────────────────────────────┐
                │     Evidence Explainability             │
                │   (Attribution + Confidence Scoring)    │
                └──────────────────┬──────────────────────┘
                                   ↓
                           Final Response
                ```

                ## 📊 Evaluation Metrics

                | Category | Metrics |
                |----------|---------|
                | Classification | Accuracy, F1 Macro, F1 Weighted, Confusion Matrix |
                | Text Generation | BLEU-1/4, ROUGE-1/2/L, BERTScore P/R/F1 |
                | RAG Quality | Fluency, Relevance, Coherence, Faithfulness |
                | Safety | Toxicity (avg/max/rate), Bias Detection |
                | Retrieval | Hit Rate @1/@5, MRR, Avg Score |
                | System | Latency (ms) |

                ## ⚠️ Disclaimer
                {MEDICAL_DISCLAIMER}
                """)

        # Footer
        gr.HTML("""
        <div style="text-align:center; color:#666; font-size:0.8em; margin-top:20px; padding:10px; 
                    border-top:1px solid #eee;">
            Medical RAG Pipeline | M.Tech Dissertation | 
            PubMedBERT + FAISS + LoRA | PubMedQA Dataset
        </div>
        """)

    return demo


def main():
    import gradio as gr
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
        css="""
            .header { text-align: center; padding: 20px; background: linear-gradient(135deg, #1565C0, #0288D1); 
                      color: white; border-radius: 12px; margin-bottom: 20px; }
            .metric-box { background: #f5f5f5; border-radius: 8px; padding: 10px; text-align: center; }
            .warning { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 8px; }
            footer { visibility: hidden; }
        """
    )


if __name__ == "__main__":
    main()