import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import axios from "axios";
import Papa from "papaparse";
import { Card } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

type DataPoint = {
  label: string;
  value: number;
};

export default function FileUploader() {
  const [csvData, setCsvData] = useState<DataPoint[]>([]);
  const [rating, setRating] = useState<string>("");
  const [downloadLink, setDownloadLink] = useState<string>("");

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type === "text/csv") {
      const formData = new FormData();
      formData.append("file", file);

      axios
        .post("http://localhost:5000/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob",
        })
        .then((response) => {
          const blob = new Blob([response.data], { type: "text/csv" });
          const url = window.URL.createObjectURL(blob);
          setDownloadLink(url);

          const reader = new FileReader();
          reader.onload = () => {
            const text = reader.result as string;
            Papa.parse<DataPoint>(text, {
              header: true,
              complete: (result) => {
                setCsvData(result.data);
              },
            });
          };
          reader.readAsText(blob);

          const ratingText = response.headers["x-rating"];
          setRating(ratingText || "Нет рейтинга");
        });
    } else {
      alert("Можно загружать только CSV файлы");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
  });

  return (
    <div>
      <Card
        {...getRootProps()}
        className="p-8 border-dashed border-2 cursor-pointer text-center"
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Отпустите файл сюда...</p>
        ) : (
          <p>Перетащите CSV файл сюда или кликните для выбора</p>
        )}
      </Card>

      {downloadLink && (
        <Button
          className="mt-4"
          onClick={() => window.open(downloadLink, "_blank")}
        >
          Скачать обработанный CSV
        </Button>
      )}

      {rating && <p className="mt-4 text-lg">Рейтинг: {rating}</p>}

      {csvData.length > 0 && (
        <div className="mt-4">
          <Chart data={csvData} />
        </div>
      )}
    </div>
  );
}

function Chart({ data }: { data: DataPoint[] }) {
  return (
    <LineChart width={600} height={300} data={data}>
      <CartesianGrid stroke="#ccc" />
      <XAxis dataKey="label" />
      <YAxis />
      <Tooltip />
      <Line type="monotone" dataKey="value" stroke="#8884d8" />
    </LineChart>
  );
}
