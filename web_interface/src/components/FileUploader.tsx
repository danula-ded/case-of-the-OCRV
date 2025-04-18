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

import { Upload } from "lucide-react";

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
    <>
      <div className="max-w-[36.146dvw] mx-auto mt-[28dvh]">
        <main className=" h-[36.633dvh]">
          <div className="p-[2.45dvh] rounded-[6.122dvh] bg-[#849030]">
            <Card
              {...getRootProps()}
              className="p-0 h-[36.6dvh] bg-[#849030] rounded-[4.59dvh] border-dashed border-2 cursor-pointer flex flex-col items-center"
            >
              <input {...getInputProps()} />
              <div className=" pt-[8.57dvh]">
                <Upload size="8dvh" color="white" />
              </div>
              {isDragActive ? (
                <p className="text-white text-[2dvh]">Отпустите файл сюда...</p>
              ) : (
                <p className="text-white  text-[2dvh]">
                  Перетащите CSV файл сюда или кликните для выбора
                </p>
              )}
              <Button className="rounded-[6.122dvh] cursor-pointer bg-white border-none text-[#849030] text-[2dvh] mb-[4.18dvh] w-[10dvw] h-[6dvh]">
                Выбрать файл
              </Button>
            </Card>
          </div>
        </main>
      </div>

      {rating && (
        <div className="mt-[5dvh] text-[4dvh] mx-auto max-w-[15dvw] flex justify-between">
          <p>Рейтинг: </p>
          <p className="text-[#529030] font-bold">{rating}</p>
        </div>
      )}

      {downloadLink && (
        <div
          className={`mx-auto ${
            rating ? "mt-[1.6dvh]" : "mt-[5dvh]"
          } max-w-fit`}
        >
          <Button
            className="rounded-[6.122dvh] border-none text-white text-[2dvh] h-[6dvh]"
            onClick={() => window.open(downloadLink, "_blank")}
          >
            Скачать обработанный CSV
          </Button>
        </div>
      )}

      {csvData.length > 0 && (
        <div className="mt-[1.6dvh] max-w-dvw">
          <Chart data={csvData} />
        </div>
      )}
    </>
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
