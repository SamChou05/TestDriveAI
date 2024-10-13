import {
  momentConfig as moments,
  type MomentType,
} from "@/config/moment-config";
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

interface VideoPlayerProps {
  src: string;
  isPlaying: boolean;
  currentTime: number;
  type: MomentType;

  onTimeUpdate?: () => void;
  onDurationChange?: () => void;
}

const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(
  (
    {
      src,
      isPlaying,
      currentTime,
      type,

      onTimeUpdate,
      onDurationChange,
    },
    ref,
  ) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [borderColor, setBorderColor] = useState<string>("transparent");

    useImperativeHandle(ref, () => videoRef.current!);

    useEffect(() => {
      if (videoRef.current) {
        if (isPlaying) {
          void videoRef.current.play();
        } else {
          videoRef.current.pause();
        }
      }
    }, [isPlaying]);

    useEffect(() => {
      if (videoRef.current) {
        videoRef.current.currentTime = currentTime;
      }
    }, [currentTime]);

    useEffect(() => {
      const currentMoment = moments.find(
        (moment) =>
          moment.type === type &&
          currentTime >= moment.startTime &&
          currentTime <= moment.endTime,
      );

      if (currentMoment) {
        setBorderColor(currentMoment.severity === "high" ? "red" : "yellow");
      } else {
        setBorderColor("transparent");
      }
    }, [currentTime, type]);

    return (
      <div
        className={`relative overflow-hidden rounded-lg ring-8 ${
          borderColor === "red"
            ? "ring-red-500"
            : borderColor === "yellow"
              ? "ring-yellow-500"
              : "ring-transparent"
        }`}
      >
        <video
          ref={videoRef}
          src={src}
          onTimeUpdate={onTimeUpdate}
          onDurationChange={onDurationChange}
          className="h-auto w-full"
        />
      </div>
    );
  },
);

VideoPlayer.displayName = "VideoPlayer";

export default VideoPlayer;
