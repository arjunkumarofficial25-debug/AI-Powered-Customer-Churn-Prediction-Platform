"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { TrendingDown } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Predict" },
  { href: "/analytics", label: "Analytics" },
  { href: "/batch", label: "Batch" },
  { href: "/model", label: "Model" },
  { href: "/sql", label: "SQL" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-10 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between md:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <TrendingDown className="size-4" aria-hidden="true" />
          </span>
          Churn Platform
        </Link>
        <nav aria-label="Main" className="flex gap-1 overflow-x-auto">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm transition-colors",
                  active ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
