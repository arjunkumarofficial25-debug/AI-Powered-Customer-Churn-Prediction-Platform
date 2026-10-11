import type { Metadata } from "next";
import Image from "next/image";
import pythonMetrics from "@/lib/generated/python-metrics.json";
import { Card, PageHeader, Stat } from "@/components/ui";
import { model } from "@/lib/model";
import { pct } from "@/lib/utils";

export const metadata: Metadata = { title: "Model & explainability | Churn Prediction Platform" };

const FIGURES = [
  { file: "shap_summary.png", title: "SHAP summary (Gradient Boosting)" },
  { file: "roc_curve.png", title: "ROC curve" },
  { file: "confusion_matrix.png", title: "Confusion matrix" },
  { file: "threshold_sweep.png", title: "Threshold sweep" },
];

export default function ModelPage() {
  const m = model.test_metrics as Record<string, number>;
  const maxImp = model.global_importance[0]?.importance ?? 1;
  const cv = Object.entries(pythonMetrics.cv_roc_auc).sort((a, b) => b[1] - a[1]);

  return (
    <>
      <PageHeader
        title="Model & explainability"
        description={`The web app scores customers with a logistic regression exported to JSON (trained on ${model.n_train.toLocaleString()} customers, tested on ${model.n_test.toLocaleString()}). The research pipeline in Python compared four models; results below.`}
      />
      <h2 className="mb-3 text-sm font-medium text-muted-foreground">Web model, held-out test set at threshold {model.threshold}</h2>
      <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-5">
        <Stat label="ROC-AUC" value={m.roc_auc.toFixed(3)} />
        <Stat label="Recall" value={pct(m.recall)} />
        <Stat label="Precision" value={pct(m.precision)} />
        <Stat label="F1" value={m.f1.toFixed(3)} />
        <Stat label="Accuracy" value={pct(m.accuracy)} />
      </div>

      <div className="mb-8 grid gap-6 md:grid-cols-2">
        <Card>
          <h2 className="mb-4 font-medium">Global feature importance</h2>
          <ul className="flex flex-col gap-2.5">
            {model.global_importance.slice(0, 10).map((f) => (
              <li key={f.feature} className="grid grid-cols-[9rem_1fr_3rem] items-center gap-3 text-sm">
                <span className="truncate">{f.feature}</span>
                <div className="h-2 rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary" style={{ width: `${(f.importance / maxImp) * 100}%` }} />
                </div>
                <span className="text-right font-mono text-xs text-muted-foreground">{f.importance.toFixed(2)}</span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-muted-foreground">Mean absolute contribution to log-odds on the test set.</p>
        </Card>
        <Card>
          <h2 className="mb-4 font-medium">Python model comparison (5-fold CV ROC-AUC)</h2>
          <table className="w-full text-sm">
            <tbody>
              {cv.map(([name, score]) => (
                <tr key={name} className="border-b border-border/60 last:border-0">
                  <td className="py-2.5">{name}{name === pythonMetrics.best_model ? <span className="ml-2 rounded-full bg-primary/15 px-2 py-0.5 text-xs text-primary">best</span> : null}</td>
                  <td className="py-2.5 text-right font-mono">{score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-xs text-muted-foreground leading-relaxed">
            Tuned Gradient Boosting: CV ROC-AUC {pythonMetrics.tuned_cv_roc_auc}, test recall {pct(pythonMetrics.test_metrics.recall)} at threshold {pythonMetrics.threshold}.
          </p>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {FIGURES.map((f) => (
          <Card key={f.file}>
            <h2 className="mb-3 font-medium">{f.title}</h2>
            <div className="overflow-hidden rounded-md bg-white">
              <Image src={`/figures/${f.file}`} alt={f.title} width={800} height={600} className="h-auto w-full" />
            </div>
          </Card>
        ))}
      </div>
    </>
  );
}
