import React from 'react';
import './TemplateView.css';

const TemplateView: React.FC = () => {
  return (
    <div className="template-view">
      <h1 className="template-view__title">Job Templates</h1>
      <div className="template-view__section">
        <h2 className="template-view__subtitle">Preset Templates</h2>
        <button className="template-view__button">
          Office Template
        </button>
      </div>
      <div className="template-view__section">
        <h2 className="template-view__subtitle">Custom Templates</h2>
        <div className="template-view__placeholder">
          <h3 className="template-view__placeholder-title">Build Your Own Template</h3>
          <p className="template-view__placeholder-text">This feature is coming soon!</p>
        </div>
      </div>
    </div>
  );
};

export default TemplateView;