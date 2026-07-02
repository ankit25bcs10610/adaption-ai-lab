import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  // spring-y press feedback (active:scale) + smooth all-property transition; keeps focus-visible ring.
  "group relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-200 ease-[cubic-bezier(0.2,0.7,0.2,1)] focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 active:scale-[0.97] cursor-pointer",
  {
    variants: {
      variant: {
        primary: "bg-run text-slate-950 hover:bg-run/90 glow-run hover:-translate-y-0.5 hover:shadow-[0_10px_30px_-8px_rgba(34,197,94,0.6)]",
        ghost: "glass hover:border-cyan/50 hover:-translate-y-0.5",
        outline: "border border-border hover:bg-muted/40 hover:border-foreground/30",
      },
      size: {
        default: "h-11 px-5 py-3",
        sm: "h-9 px-4",
        lg: "h-12 px-6 text-base",
      },
    },
    defaultVariants: { variant: "primary", size: "default" },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
