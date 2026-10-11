import type { Metadata } from "next";
import { BatchPredictor } from "@/components/batch-predictor";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Batch prediction | Churn Prediction Platform" };

export default function BatchPage() {
  return (
    <>
      <PageHeader title="Batch prediction" description="Upload a CSV of customers to score them all at once. Columns are validated, results are ranked by churn risk, and you can download the scored file." />
      <BatchPredictor />
    </>
  );
}
