import { type Moment } from "@/config/moment-config";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export const MomentDialog: React.FC<{
  moment: Moment;
  duration: number;
  onSeek: (time: number) => void;
}> = ({ moment, duration, onSeek }) => {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering parent elements' click events
    onSeek(moment.startTime);
  };

  return (
    <TooltipProvider>
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          <div
            className="absolute top-1/2 -translate-y-[7px] hover:cursor-pointer"
            style={{
              left: `calc(${(moment.startTime / duration) * 100}% - 2px)`,
            }}
            onClick={handleClick}
          >
            <div
              className={`h-4 w-4 rounded-full ${
                moment.severity === "low" ? "bg-yellow-400" : "bg-red-500"
              }`}
            />
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{moment.content}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
