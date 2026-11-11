import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';
import { DatasetSummary } from '../types';

export interface AppNotification {
  id: string;
  title: string;
  message: string;
  timestamp: string;
  type: 'info' | 'success' | 'warning' | 'alert';
  read: boolean;
}

export interface AppSettings {
  showReferenceLines: boolean;
  animateCharts: boolean;
  pageSize: number;
  alertsEnabled: boolean;
}

interface AppContextType {
  // Dataset state
  datasetSummary: DatasetSummary | null;
  refreshDatasetSummary: () => Promise<void>;
  
  // Traffic Filter
  activeFilter: string; // 'All' | 'Volumetric' | 'Rates' | 'Temporal'
  setActiveFilter: (filter: string) => void;

  // Search
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  searchResults: any[];
  searchLoading: boolean;
  performSearch: (q: string) => Promise<void>;
  clearSearch: () => void;

  // Notifications
  notifications: AppNotification[];
  unreadCount: number;
  markAllNotificationsRead: () => void;
  clearNotifications: () => void;

  // Settings
  settings: AppSettings;
  updateSettings: (newSettings: Partial<AppSettings>) => void;
  resetSettings: () => void;

  // Dialog / Modal toggles
  isNotificationsOpen: boolean;
  setIsNotificationsOpen: (open: boolean) => void;
  isSettingsOpen: boolean;
  setIsSettingsOpen: (open: boolean) => void;
  isProfileOpen: boolean;
  setIsProfileOpen: (open: boolean) => void;
}

const defaultSettings: AppSettings = {
  showReferenceLines: true,
  animateCharts: true,
  pageSize: 25,
  alertsEnabled: true,
};

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [activeFilter, setActiveFilter] = useState<string>('All');
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  // Modals state
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  // Settings with LocalStorage persistence
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = localStorage.getItem('netguard_settings');
      return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    } catch {
      return defaultSettings;
    }
  });

  const updateSettings = (partial: Partial<AppSettings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...partial };
      try {
        localStorage.setItem('netguard_settings', JSON.stringify(updated));
      } catch (e) {
        console.error(e);
      }
      return updated;
    });
  };

  const resetSettings = () => {
    setSettings(defaultSettings);
    try {
      localStorage.removeItem('netguard_settings');
    } catch (e) {
      console.error(e);
    }
  };

  // Real Application Notifications (no fabricated security alerts)
  const [notifications, setNotifications] = useState<AppNotification[]>([
    {
      id: 'notif-1',
      title: 'Dataset Ingestion Verified',
      message: 'Benchmark multi_signal_raw.csv loaded (32,700 flows across 5 signal families).',
      timestamp: '10:00 AM',
      type: 'success',
      read: false,
    },
    {
      id: 'notif-2',
      title: 'Deep Autoencoder Ready',
      message: 'PyTorch architecture (78→128→64→16→64→128→78) calibrated with validation Median/MAD.',
      timestamp: '10:05 AM',
      type: 'info',
      read: false,
    },
    {
      id: 'notif-3',
      title: 'Stabilized Adaptive Threshold Active',
      message: 'Bounded clamping [0.7τ₀, 1.3τ₀] with exponential smoothing α=0.05 protecting streaming buffer.',
      timestamp: '10:15 AM',
      type: 'info',
      read: false,
    },
  ]);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAllNotificationsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const clearNotifications = () => {
    setNotifications([]);
  };

  const refreshDatasetSummary = async () => {
    try {
      const sum = await api.getDatasetSummary();
      setDatasetSummary(sum);
    } catch (e) {
      console.error('Failed to load dataset summary:', e);
    }
  };

  useEffect(() => {
    refreshDatasetSummary();
  }, []);

  const performSearch = async (q: string) => {
    const term = q.trim();
    setSearchQuery(term);
    if (!term) {
      setSearchResults([]);
      return;
    }
    try {
      setSearchLoading(true);
      const res = await api.searchDataset(term, 20);
      setSearchResults(res.results || []);
    } catch (e) {
      console.error('Search query failed:', e);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
  };

  return (
    <AppContext.Provider
      value={{
        datasetSummary,
        refreshDatasetSummary,
        activeFilter,
        setActiveFilter,
        searchQuery,
        setSearchQuery,
        searchResults,
        searchLoading,
        performSearch,
        clearSearch,
        notifications,
        unreadCount,
        markAllNotificationsRead,
        clearNotifications,
        settings,
        updateSettings,
        resetSettings,
        isNotificationsOpen,
        setIsNotificationsOpen,
        isSettingsOpen,
        setIsSettingsOpen,
        isProfileOpen,
        setIsProfileOpen,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
