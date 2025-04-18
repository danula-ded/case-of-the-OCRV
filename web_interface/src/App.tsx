import { Header } from "./components/Header";
import FileUploader from "./components/FileUploader";

function App() {
  return (
    <div className=" bg-[url('./assets/bg.svg')] bg-cover bg-center w-dvw h-dvh">
      <Header />
      <div className="p-8 max-w-4xl mx-auto">
        <main className="p-4">
          <FileUploader />
        </main>
      </div>
    </div>
  );
}

export default App;
