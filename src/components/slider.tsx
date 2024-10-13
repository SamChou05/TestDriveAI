import React, { useState, useEffect } from "react";
import { momentConfig } from "@/config/moment-config";
import { MomentDialog } from "./moment-dialog";

interface SliderProps {
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

const Slider: React.FC<SliderProps> = ({ duration, currentTime, onSeek }) => {
  const [sliderValue, setSliderValue] = useState(0);

  useEffect(() => {
    setSliderValue(currentTime);
  }, [currentTime]);

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value);
    setSliderValue(newValue);
    onSeek(newValue);
  };

  return (
    <div className="flex w-full items-center space-x-4">
      <span className="w-8 text-sm font-medium">{formatTime(currentTime)}</span>
      <div className="relative flex-grow">
        <input
          type="range"
          min={0}
          max={duration}
          value={sliderValue}
          onChange={handleSliderChange}
          className="slider-thumb relative z-10 h-2 w-full cursor-pointer appearance-none rounded-full bg-gray-200"
          style={{
            backgroundSize: `${(sliderValue / duration) * 100}% 100%`,
            backgroundImage: `linear-gradient(#3b82f6, #3b82f6)`,
            backgroundRepeat: "no-repeat",
          }}
        />
        <div className="absolute inset-0 z-20">
          {momentConfig.map((moment) => (
            <MomentDialog
              key={moment.startTime}
              moment={moment}
              duration={duration}
              onSeek={onSeek}
            />
          ))}
        </div>
      </div>
      <span className="text-sm font-medium">{formatTime(duration)}</span>
    </div>
  );
};

const formatTime = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
};

export default Slider;
