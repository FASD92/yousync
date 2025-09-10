import React from 'react';

const Loader = () => {
  return (
    <div className="flex items-center justify-center">
      <div className="animate-spin w-8 h-8 border-3 border-green-400 border-t-transparent rounded-full" />
    </div>
  );
};

export default Loader; 