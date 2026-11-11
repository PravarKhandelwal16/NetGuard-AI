import React from 'react';
import { AppProvider } from './context/AppContext';
import { SinglePageDashboard } from './pages/SinglePageDashboard';

export const App: React.FC = () => {
  return (
    <AppProvider>
      <SinglePageDashboard />
    </AppProvider>
  );
};

export default App;
