import { Header } from "./components/Header";
import FileUploader from "./components/FileUploader";

function App() {
  return (
    <div className=" bg-[url('./assets/bg.svg')] bg-cover bg-center w-dvw h-dvh">
      <Header />

      <FileUploader />
    </div>
  );
}

export default App;
