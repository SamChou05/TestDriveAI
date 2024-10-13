"use client";

import { CloudUpload } from "lucide-react";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { type ReactNode } from "react";

const MyDropzone = ({ message }: { message: ReactNode }) => {
  const [file, setFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div
      {...getRootProps()}
      className="flex h-80 w-full cursor-pointer items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/20 p-8 text-center transition-colors hover:border-muted-foreground/40"
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center justify-center">
        {file ? (
          <>
            <p className="text-lg font-medium text-muted-foreground">
              File added:
            </p>
            <p className="mt-2 text-sm font-bold text-muted-foreground">
              {file.name}
            </p>
          </>
        ) : (
          <>
            <CloudUpload className="h-10 w-10 text-muted-foreground" />
            {isDragActive ? (
              <p className="text-lg font-medium text-muted-foreground">
                Drop your video here ...
              </p>
            ) : (
              <p className="text-lg font-medium text-muted-foreground">
                {message}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export const Dropzones = () => {
  return (
    <div className="grid w-full max-w-4xl grid-cols-1 gap-10 md:grid-cols-2">
      <MyDropzone
        message={
          <>
            Add your <span className="font-bold italic">dash cam</span> footage
            here
          </>
        }
      />
      <MyDropzone
        message={
          <>
            Add your <span className="font-bold italic">driver</span> footage
            here
          </>
        }
      />
    </div>
  );
};
