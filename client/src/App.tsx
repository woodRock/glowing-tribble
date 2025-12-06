import { useState } from 'react';
import HeroSelect from './views/HeroSelect';
import TemplateView from './views/TemplateView';
import StaffManagement from './components/StaffManagement'; // Import StaffManagement
import './App.css';

type View = 'roster' | 'templates' | 'staff'; // Add 'staff' view

function App() {
  const [view, setView] = useState<View>('roster');

  return (
    <div className="app">
      <nav className="app__nav">
        <button
          onClick={() => setView('roster')}
          className={`app__nav-button ${view === 'roster' ? 'app__nav-button--active' : ''}`}
        >
          Roster
        </button>
        <button
          onClick={() => setView('templates')}
          className={`app__nav-button ${view === 'templates' ? 'app__nav-button--active' : ''}`}
        >
          Templates
        </button>
        <button
          onClick={() => setView('staff')} // New button for Staff Management
          className={`app__nav-button ${view === 'staff' ? 'app__nav-button--active' : ''}`}
        >
          Staff
        </button>
      </nav>
      <main className="app__main">
        {view === 'roster' && <HeroSelect />}
        {view === 'templates' && <TemplateView />}
        {view === 'staff' && <StaffManagement />} {/* Render StaffManagement */}
      </main>
    </div>
  );
}

export default App;