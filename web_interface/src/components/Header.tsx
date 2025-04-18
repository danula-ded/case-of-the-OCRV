import mainlogo from "@/assets/logo.svg";
import name from "@/assets/ОЦРВ.svg";

export const Header = () => {
  return (
    <header className="flex items-center justify-between px-[4.5dvw] bg-[#242424] h-[9dvh] shadow-lg">
      <img src={mainlogo} alt="Логотип" className="h-[9dvh]" />
      <img src={name} alt="ОЦРВ" className="h-[5dvh]" />
    </header>
  );
};
