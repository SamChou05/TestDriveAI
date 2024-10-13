import { Dropzones } from "@/components/dropzones";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen w-full flex-col items-center px-4 pt-36">
      {/* <Image
        src="/logo.png"
        alt="logo"
        width={400}
        height={160}
        className="mb-16"
      /> */}
      <Dropzones />
      <Link
        className={cn(buttonVariants(), "mt-10 px-8 shadow-md")}
        href="/analyze"
      >
        Analyze Footage
      </Link>
    </div>
  );
}
