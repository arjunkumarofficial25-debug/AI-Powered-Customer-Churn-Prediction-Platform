import { PredictionForm } from "@/components/prediction-form";
import { PageHeader } from "@/components/ui";

export default function Page() {
  return (
    <>
      <PageHeader
        title="Single customer prediction"
        description="Adjust a customer profile to see their churn risk, the factors driving it, and suggested retention actions. Scoring runs instantly in your browser."
      />
      <PredictionForm />
    </>
  );
}
