import hakaton from "@/assets/logo.svg";
import mainlogo from "@/assets/TEAzaurus.svg";

export const Header = () => {
  return (
    <header className="flex items-center justify-between px-[4.5dvw] bg-[#242424] h-[9dvh] shadow-lg">
      <img src={mainlogo} alt="Логотип" className="h-[9dvh]" />
      <img src={hakaton} alt="Логотип" className="h-[9dvh]" />
    </header>
  );
};
