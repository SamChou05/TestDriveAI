import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { AvatarImage } from "@radix-ui/react-avatar";

export function Navbar() {
  return (
    <nav className="fixed left-0 right-0 top-4 z-50">
      <div className="container mx-auto px-4">
        <div className="relative flex items-center justify-center">
          <div className="flex max-w-fit items-center justify-center space-x-2 rounded-full border bg-background/90 px-4 py-2 shadow-sm backdrop-blur-sm">
            <NavLink href="/" label="Home" />
            <NavLink href="/analyze" label="Learn" />
            <NavLink href="/analyze" label="Past Drives" />
            <NavLink href="/report" label="Reports" />
            <NavLink href="/settings" label="Settings" />
          </div>
        </div>
      </div>
      <div className="absolute right-4 top-1/2 -translate-y-1/2">
        <Avatar className="h-10 w-10">
          <AvatarImage src="/isaac.jpg" />
          <AvatarFallback>ID</AvatarFallback>
        </Avatar>
      </div>
    </nav>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className={cn(
        buttonVariants({ variant: "ghost", size: "sm" }),
        "rounded-full text-sm font-medium transition-colors hover:bg-accent",
      )}
    >
      {label}
    </Link>
  );
}
