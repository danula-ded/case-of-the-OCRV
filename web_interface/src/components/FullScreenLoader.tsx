import React from "react";

const FullScreenLoader: React.FC = () => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-opacity-50 backdrop-blur-sm">
      <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
    </div>
  );
};

export default FullScreenLoader;
