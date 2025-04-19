import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend,
} from "recharts";

type DataPoint = {
  time: string;
  incident: number;
};

type Props = {
  data: DataPoint[];
};

const COLORS = ["#529030", "#e63946"];

export function Charts({ data }: Props) {
  // Соотношение 0 и 1
  const counts = data.reduce(
    (acc, cur) => {
      if (cur.incident === 0) acc.zeros += 1;
      else acc.ones += 1;
      return acc;
    },
    { zeros: 0, ones: 0 }
  );

  const pieData = [
    { name: "Нет инцидента (0)", value: counts.zeros },
    { name: "Инцидент (1)", value: counts.ones },
  ];

  // Барчарт по часам
  const hoursData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i.toString().padStart(2, "0")}:00`,
    incidents: data.filter(
      (d) => d.incident === 1 && new Date(d.time).getHours() === i
    ).length,
  }));

  // Линейный график по дням
  const dailyStats: { date: string; zeros: number; ones: number }[] = [];
  data.forEach((d) => {
    const date = d.time.split(" ")[0];
    const existing = dailyStats.find((stat) => stat.date === date);
    if (existing) {
      if (d.incident === 0) existing.zeros++;
      else existing.ones++;
    } else {
      dailyStats.push({
        date,
        zeros: d.incident === 0 ? 1 : 0,
        ones: d.incident === 1 ? 1 : 0,
      });
    }
  });

  return (
    <div className="flex flex-col gap-[4dvh] items-center">
      {/* PieChart */}
      <PieChart width={400} height={300}>
        <Pie
          data={pieData}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={100}
          label
        >
          {pieData.map((_, index) => (
            <Cell key={index} fill={COLORS[index]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>

      {/* BarChart по часам */}
      <BarChart width={600} height={300} data={hoursData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="hour" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="incidents" fill="#e63946" />
      </BarChart>

      {/* LineChart по дням */}
      <LineChart width={600} height={300} data={dailyStats}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="zeros" stroke="#529030" name="Нули" />
        <Line type="monotone" dataKey="ones" stroke="#e63946" name="Единицы" />
      </LineChart>
    </div>
  );
}
