import { useState } from 'react';
import HeroSelect from './views/HeroSelect';
import './App.css';

function App() {
  return (
    <div className="app">
      <main className="app__main">
        <HeroSelect />
      </main>
    </div>
  );
}

export default App;