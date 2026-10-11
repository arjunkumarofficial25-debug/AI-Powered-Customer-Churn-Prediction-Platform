"use client";

import { useState } from "react";
import queries from "@/lib/generated/queries.json";
import { Card } from "@/components/ui";
import { cn } from "@/lib/utils";

export function SqlExplorer() {
  const [active, setActive] = useState(0);
  const q = queries[active];

  return (
    <div className="grid gap-6 lg:grid-cols-[16rem_1fr]">
      <nav aria-label="Saved queries" className="flex flex-col gap-1">
        {queries.map((item, i) => (
          <button
            key={item.title}
            type="button"
            onClick={() => setActive(i)}
            aria-pressed={i === active}
            className={cn(
              "rounded-md px-3 py-2 text-left text-sm transition-colors",
              i === active ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
            )}
          >
            {item.title}
          </button>
        ))}
      </nav>
      <div className="flex min-w-0 flex-col gap-4">
        <Card className="p-0">
          <pre className="overflow-x-auto p-4 font-mono text-xs leading-relaxed text-primary">{q.sql}</pre>
        </Card>
        <Card className="overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead className="border-b border-border text-left text-xs text-muted-foreground">
              <tr>
                {q.columns.map((c) => <th key={c} scope="col" className="px-4 py-3 font-medium">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {q.rows.map((row, i) => (
                <tr key={i} className="border-b border-border/60 last:border-0">
                  {row.map((cell, j) => (
                    <td key={j} className={cn("px-4 py-2.5", typeof cell === "number" && "font-mono")}>{String(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-border px-4 py-2 text-xs text-muted-foreground">{q.rows.length} rows</p>
        </Card>
      </div>
    </div>
  );
}
