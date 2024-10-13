import { Car } from "lucide-react";
import React, { useEffect, useRef } from "react";

export const Loading: React.FC = () => {
  const carRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const car = carRef.current;
    const container = containerRef.current;
    if (!car || !container) return;

    let start: number | null = null;
    const duration = 9000; // 16 seconds in milliseconds

    function animate(timestamp: number) {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const normalizedProgress = progress / duration;

      // Use sine function to create a speed variation without reversing
      const speedFactor =
        Math.sin(normalizedProgress * Math.PI * 2) * 0.5 + 1.5;

      // Calculate the position based on the integral of the speed factor
      const position =
        normalizedProgress +
        (Math.cos(normalizedProgress * Math.PI * 2) - 1) / (Math.PI * 4);

      const maxTranslation =
        (container?.offsetWidth ?? 0) - (car?.offsetWidth ?? 0);
      const adjustedPosition = (position % 1) * maxTranslation;

      car.style.transform = `translateX(${adjustedPosition}px)`;

      if (progress < duration) {
        requestAnimationFrame(animate);
      } else {
        // Reset the animation
        start = null;
        requestAnimationFrame(animate);
      }
    }

    requestAnimationFrame(animate);

    // Cleanup function
    return () => {
      start = null;
    };
  }, []);

  return (
    <div className="flex h-screen items-center justify-center">
      <div ref={containerRef} className="relative h-20 w-full max-w-md">
        <div className="absolute bottom-0 left-0 right-0 h-3 bg-foreground"></div>
        <div ref={carRef} className="absolute bottom-1 left-0">
          <Car className="h-12 w-12 text-foreground" />
        </div>
      </div>
    </div>
  );
};
