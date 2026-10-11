import { cn } from "@/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-lg border border-border bg-card p-5", className)} {...props} />;
}

export function PageHeader({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-6 flex flex-col gap-1">
      <h1 className="text-2xl font-semibold tracking-tight text-balance">{title}</h1>
      <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground text-pretty">{description}</p>
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card className="flex flex-col gap-1 p-4">
      <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="font-mono text-2xl font-semibold">{value}</span>
      {hint ? <span className="text-xs text-muted-foreground">{hint}</span> : null}
    </Card>
  );
}
