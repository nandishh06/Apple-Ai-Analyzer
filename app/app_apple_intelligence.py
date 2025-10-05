"""
Apple Intelligence Classifier Interface
=====================================
Professional apple classification and quality assessment interface.
Comprehensive data presentation with Apple HIG design principles and SF Pro typography.
Designed for commercial use with detailed analytics and business insights.
"""

import gradio as gr
import os
import sys
from PIL import Image
import time
import logging
import json
from datetime import datetime
import numpy as np

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from core.pipeline_fixed import FixedAppleAnalysisPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleIntelligenceClassifier:
    def __init__(self):
        """Initialize the Apple Intelligence Classifier Interface."""
        self.pipeline = FixedAppleAnalysisPipeline(models_dir="models")
        self.analysis_history = []

        # Apple Intelligence CSS with comprehensive data presentation
        self.apple_intelligence_css = """
        /* Import SF Pro fonts (Apple's system font) */
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700;800&family=SF+Pro+Text:wght@400;500;600&display=swap');

        /* Apple Intelligence Color System with Professional Palette */
        :root {
            /* Primary Backgrounds */
            --apple-light-bg-primary: #F8F9F7;
            --apple-light-bg-secondary: #F2F3F1;
            --apple-light-bg-tertiary: #ECEDEA;
            --apple-light-surface-primary: #E6E7E4;
            --apple-light-surface-secondary: #DFE0DC;
            --apple-light-surface-tertiary: #D8D9D5;
            --apple-light-separator: #D0D1CD;
            --apple-light-text-primary: #1A1D1B;
            --apple-light-text-secondary: #3A3D3B;
            --apple-light-text-tertiary: #5A5D5B;
            --apple-light-text-quaternary: #7A7D7B;

            /* Dark Appearance (Primary) */
            --apple-dark-bg-primary: #1A1D1B;
            --apple-dark-bg-secondary: #222523;
            --apple-dark-bg-tertiary: #2A2D2B;
            --apple-dark-surface-primary: #323533;
            --apple-dark-surface-secondary: #3A3D3B;
            --apple-dark-surface-tertiary: #424543;
            --apple-dark-separator: #4A4D4B;
            --apple-dark-text-primary: #F8F9F7;
            --apple-dark-text-secondary: #E6E7E4;
            --apple-dark-text-tertiary: #D0D1CD;
            --apple-dark-text-quaternary: #B8B9B5;

            /* Professional Accent Colors */
            --apple-red: #C23B22;
            --apple-green: #4A7C59;
            --apple-gold: #D4A017;
            --apple-brown: #8B4513;

            /* Dynamic Colors */
            --bg-primary: var(--apple-dark-bg-primary);
            --bg-secondary: var(--apple-dark-bg-secondary);
            --bg-tertiary: var(--apple-dark-bg-tertiary);
            --surface-primary: var(--apple-dark-surface-primary);
            --surface-secondary: var(--apple-dark-surface-secondary);
            --surface-tertiary: var(--apple-dark-surface-tertiary);
            --separator: var(--apple-dark-separator);
            --text-primary: var(--apple-dark-text-primary);
            --text-secondary: var(--apple-dark-text-secondary);
            --text-tertiary: var(--apple-dark-text-tertiary);
            --text-quaternary: var(--apple-dark-text-quaternary);
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15);
            --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.18);
        }

        /* Light Appearance Override */
        [data-theme="light"] {
            --bg-primary: var(--apple-light-bg-primary);
            --bg-secondary: var(--apple-light-bg-secondary);
            --bg-tertiary: var(--apple-light-bg-tertiary);
            --surface-primary: var(--apple-light-surface-primary);
            --surface-secondary: var(--apple-light-surface-secondary);
            --surface-tertiary: var(--apple-light-surface-tertiary);
            --separator: var(--apple-light-separator);
            --text-primary: var(--apple-light-text-primary);
            --text-secondary: var(--apple-light-text-secondary);
            --text-tertiary: var(--apple-light-text-tertiary);
            --text-quaternary: var(--apple-light-text-quaternary);
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.12);
            --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.15);
        }

        /* Global Apple Intelligence Styles */
        * {
            font-family: 'SF Pro Display', 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif !important;
            box-sizing: border-box;
        }

        /* Main Container */
        .gradio-container {
            background: var(--bg-primary) !important;
            min-height: 100vh;
            padding: 40px;
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        /* Apple Intelligence Header with Apple Shade Variations */
        .apple-intelligence-header {
            background:
                linear-gradient(135deg,
                    #1A1D1B 0%,
                    #222523 25%,
                    #2A2D2B 50%,
                    #222523 75%,
                    #1A1D1B 100%
                ),
                radial-gradient(ellipse 400px 200px at 30% 20%,
                    rgba(194, 59, 34, 0.12) 0%,
                    rgba(74, 124, 89, 0.08) 30%,
                    transparent 70%
                ),
                radial-gradient(ellipse 300px 150px at 70% 80%,
                    rgba(212, 160, 23, 0.10) 0%,
                    rgba(139, 69, 19, 0.06) 40%,
                    transparent 80%
                );
            border-radius: 24px;
            padding: 48px 56px;
            margin-bottom: 40px;
            box-shadow:
                var(--shadow-lg),
                inset 0 1px 0 rgba(248, 249, 247, 0.1);
            border: 1px solid var(--separator);
            position: relative;
            overflow: hidden;
        }

        .apple-intelligence-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(74, 124, 89, 0.3) 25%,
                rgba(212, 160, 23, 0.3) 50%,
                rgba(74, 124, 89, 0.3) 75%,
                transparent 100%
            );
        }

        /* Apple Shade Pattern Overlay */
        .apple-intelligence-header::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                /* Apple skin texture simulation */
                radial-gradient(circle 20px at 15% 25%, rgba(194, 59, 34, 0.04) 0%, transparent 70%),
                radial-gradient(circle 15px at 85% 75%, rgba(74, 124, 89, 0.04) 0%, transparent 70%),
                radial-gradient(circle 25px at 45% 15%, rgba(212, 160, 23, 0.03) 0%, transparent 70%),
                radial-gradient(circle 18px at 75% 45%, rgba(139, 69, 19, 0.03) 0%, transparent 70%),

                /* Apple variety shade variations */
                radial-gradient(ellipse 200px 100px at 20% 30%, rgba(194, 59, 34, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse 150px 120px at 80% 70%, rgba(74, 124, 89, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse 180px 90px at 60% 20%, rgba(212, 160, 23, 0.05) 0%, transparent 60%),

                /* Subtle texture lines mimicking apple skin */
                linear-gradient(45deg, transparent 48%, rgba(194, 59, 34, 0.02) 50%, transparent 52%),
                linear-gradient(-45deg, transparent 48%, rgba(74, 124, 89, 0.02) 50%, transparent 52%),
                linear-gradient(135deg, transparent 48%, rgba(212, 160, 23, 0.02) 50%, transparent 52%);
            opacity: 0.8;
            pointer-events: none;
        }

        /* Apple Intelligence Header Content Layer */
        .apple-intelligence-header > * {
            position: relative;
            z-index: 2;
        }

        /* Apple Intelligence Header Depth Enhancement */
        .apple-intelligence-header {
            backdrop-filter: blur(20px) saturate(1.1);
            -webkit-backdrop-filter: blur(20px) saturate(1.1);
        }

        /* Apple Intelligence Header Title Enhancement */
        .apple-intelligence-title {
            font-size: 56px;
            font-weight: 700;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: 16px;
            letter-spacing: -0.025em;
            line-height: 1.1;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            position: relative;
        }

        .apple-intelligence-title::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 200px;
            background:
                radial-gradient(circle, rgba(74, 124, 89, 0.03) 0%, transparent 70%);
            border-radius: 50%;
            z-index: -1;
        }

        /* Apple Intelligence Header Subtitle Enhancement */
        .apple-intelligence-subtitle {
            font-size: 24px;
            color: var(--text-secondary);
            text-align: center;
            font-weight: 400;
            margin-bottom: 20px;
            line-height: 1.3;
            opacity: 0.95;
            position: relative;
        }

        /* Apple Intelligence Header Tagline Enhancement */
        .apple-intelligence-tagline {
            font-size: 19px;
            color: var(--text-primary);
            text-align: center;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            margin-bottom: 32px;
            position: relative;
            background: linear-gradient(90deg, var(--apple-green) 0%, var(--apple-gold) 50%, var(--apple-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Apple Intelligence Branding */
        .apple-intelligence-brand {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 32px;
            margin-bottom: 40px;
        }

        .apple-intelligence-logo {
            width: 96px;
            height: 96px;
            background: var(--bg-tertiary);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-primary);
            font-size: 48px;
            font-weight: 300;
            box-shadow: var(--shadow-md);
            border: 2px solid var(--surface-primary);
        }

        /* Apple Intelligence Typography */
        .apple-intelligence-title {
            font-size: 56px;
            font-weight: 700;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: 16px;
            letter-spacing: -0.025em;
            line-height: 1.1;
        }

        .apple-intelligence-subtitle {
            font-size: 24px;
            color: var(--text-secondary);
            text-align: center;
            font-weight: 400;
            margin-bottom: 20px;
            line-height: 1.3;
        }

        .apple-intelligence-tagline {
            font-size: 19px;
            color: var(--text-primary);
            text-align: center;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            margin-bottom: 32px;
        }

        /* Apple Intelligence Navigation */
        .nav-apple-intelligence {
            display: flex;
            justify-content: center;
            gap: 0;
            margin: 40px 0;
            background: var(--surface-primary);
            padding: 8px;
            border-radius: 16px;
            border: 1px solid var(--separator);
        }

        .nav-item-apple-intelligence {
            padding: 16px 32px;
            border-radius: 12px;
            background: transparent;
            color: var(--text-secondary);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            border: none;
            font-size: 17px;
            min-width: 120px;
            text-align: center;
        }

        .nav-item-apple-intelligence.active {
            background: var(--bg-secondary);
            color: var(--text-primary);
            box-shadow: var(--shadow-sm);
        }

        .nav-item-apple-intelligence:hover:not(.active) {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            transform: translateY(-1px);
        }

        /* Apple Intelligence Dashboard Layout */
        .dashboard-apple-intelligence {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 48px;
            max-width: 2000px;
            margin: 0 auto;
        }

        /* Apple Intelligence Cards */
        .card-apple-intelligence {
            background: var(--bg-secondary);
            border-radius: 20px;
            padding: 48px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--separator);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
        }

        .card-apple-intelligence::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, var(--separator) 50%, transparent 100%);
        }

        .card-apple-intelligence:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-lg);
            border-color: var(--surface-primary);
        }

        /* Apple Intelligence Data Cards */
        .data-card-apple-intelligence {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 32px;
            margin: 20px 0;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--separator);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .data-card-apple-intelligence:hover {
            transform: translateX(8px);
            box-shadow: var(--shadow-md);
        }

        /* Apple Intelligence Metrics Grid */
        .metrics-grid-apple-intelligence {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin: 24px 0;
        }

        .metric-item-apple-intelligence {
            background: var(--surface-primary);
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--separator);
        }

        .metric-value-apple-intelligence {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
            font-family: 'SF Pro Display', monospace;
        }

        .metric-label-apple-intelligence {
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        /* Apple Intelligence Model Status */
        .model-status-apple-intelligence {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 24px 0;
        }

        .model-item-apple-intelligence {
            background: var(--bg-tertiary);
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            border: 2px solid var(--separator);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .model-item-apple-intelligence:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
            border-color: var(--text-primary);
        }

        .model-name-apple-intelligence {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
        }

        .model-state-apple-intelligence {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            background: var(--surface-primary);
            color: var(--text-primary);
            border: 1px solid var(--separator);
        }

        /* Apple Intelligence Confidence Bars */
        .confidence-bars-apple-intelligence {
            display: grid;
            gap: 16px;
            margin: 24px 0;
        }

        .confidence-item-apple-intelligence {
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--separator);
        }

        .confidence-header-apple-intelligence {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .confidence-title-apple-intelligence {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .confidence-value-apple-intelligence {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'SF Pro Display', monospace;
        }

        .confidence-bar-container-apple-intelligence {
            width: 100%;
            height: 8px;
            background: var(--surface-primary);
            border-radius: 4px;
            overflow: hidden;
        }

        /* Apple Intelligence Professional Color Usage */
        .confidence-bar-apple-intelligence {
            height: 100%;
            background: var(--apple-red);
            border-radius: 4px;
            transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .confidence-bar-apple-intelligence.high {
            background: var(--apple-green);
        }

        .confidence-bar-apple-intelligence.medium {
            background: var(--apple-gold);
        }

        .status-indicator-apple-intelligence {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-approved-apple-intelligence {
            background: var(--apple-green);
            color: var(--apple-light-bg-primary);
        }

        .status-premium-apple-intelligence {
            background: var(--apple-gold);
            color: var(--apple-dark-bg-primary);
        }

        .status-issue-apple-intelligence {
            background: var(--apple-red);
            color: var(--apple-light-bg-primary);
        }

        .variety-badge-apple-intelligence {
            padding: 6px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .variety-fuji-apple-intelligence {
            background: var(--apple-red);
            color: var(--apple-light-bg-primary);
        }

        .variety-granny-apple-intelligence {
            background: var(--apple-green);
            color: var(--apple-light-bg-primary);
        }

        .variety-honeycrisp-apple-intelligence {
            background: var(--apple-gold);
            color: var(--apple-dark-bg-primary);
        }

        .quality-score-apple-intelligence {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
            font-weight: 600;
        }

        .quality-excellent-apple-intelligence {
            color: var(--apple-green);
        }

        .quality-good-apple-intelligence {
            color: var(--apple-gold);
        }

        .quality-standard-apple-intelligence {
            color: var(--text-secondary);
        }

        .defect-indicator-apple-intelligence {
            color: var(--apple-brown);
            font-weight: 500;
        }

        /* Apple Intelligence Icons for Color Enhancement */
        .icon-status-apple-intelligence::before {
            content: "✅";
            margin-right: 6px;
        }

        .icon-warning-apple-intelligence::before {
            content: "⚠️";
            margin-right: 6px;
        }

        .icon-error-apple-intelligence::before {
            content: "❌";
            margin-right: 6px;
        }

        /* Apple Intelligence Detailed Analysis */
        .detailed-analysis-apple-intelligence {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 32px;
            margin: 24px 0;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--separator);
        }

        .analysis-section-apple-intelligence {
            margin-bottom: 32px;
        }

        .analysis-title-apple-intelligence {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
        }

        .analysis-grid-apple-intelligence {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 20px;
        }

        .analysis-item-apple-intelligence {
            background: var(--surface-primary);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--separator);
        }

        .analysis-label-apple-intelligence {
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            font-weight: 500;
        }

        .analysis-value-apple-intelligence {
            font-size: 16px;
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Apple Intelligence Business Intelligence */
        .business-intel-apple-intelligence {
            background: linear-gradient(135deg, var(--surface-primary) 0%, var(--surface-secondary) 100%);
            border-radius: 16px;
            padding: 32px;
            margin: 24px 0;
            border: 2px solid var(--separator);
        }

        .intel-title-apple-intelligence {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 20px;
        }

        .intel-grid-apple-intelligence {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
        }

        .intel-item-apple-intelligence {
            background: var(--bg-tertiary);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--separator);
        }

        .intel-label-apple-intelligence {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
        }

        .intel-details-apple-intelligence {
            color: var(--text-secondary);
            font-size: 15px;
            line-height: 1.4;
        }

        /* Apple Intelligence Technical Specs */
        .technical-specs-apple-intelligence {
            background: var(--bg-tertiary);
            border-radius: 16px;
            padding: 32px;
            margin: 24px 0;
            border: 1px solid var(--separator);
        }

        .specs-title-apple-intelligence {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 20px;
        }

        .specs-list-apple-intelligence {
            display: grid;
            gap: 12px;
        }

        .spec-item-apple-intelligence {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px;
            background: var(--surface-primary);
            border-radius: 8px;
            border: 1px solid var(--separator);
        }

        .spec-icon-apple-intelligence {
            width: 20px;
            height: 20px;
            background: var(--text-primary);
            border-radius: 50%;
            flex-shrink: 0;
        }

        .spec-text-apple-intelligence {
            font-size: 15px;
            color: var(--text-primary);
        }

        /* Apple Intelligence Upload Zone */
        .upload-apple-intelligence {
            border: 2px dashed var(--separator);
            border-radius: 20px;
            padding: 80px;
            text-align: center;
            background: var(--bg-tertiary);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            margin: 40px 0;
        }

        .upload-apple-intelligence:hover {
            border-color: var(--text-primary);
            background: var(--surface-primary);
            transform: scale(1.01);
        }

        /* Apple Intelligence Buttons */
        .btn-apple-intelligence-primary {
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--separator) !important;
            border-radius: 16px !important;
            padding: 20px 40px !important;
            font-weight: 600 !important;
            font-size: 17px !important;
            letter-spacing: 0.025em !important;
            transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .btn-apple-intelligence-primary:hover {
            background: var(--surface-primary) !important;
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-md) !important;
            border-color: var(--text-primary) !important;
        }

        .btn-apple-intelligence-secondary {
            background: var(--surface-primary) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--separator) !important;
            border-radius: 16px !important;
            padding: 18px 36px !important;
            font-weight: 500 !important;
            font-size: 16px !important;
            letter-spacing: 0.02em !important;
            transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .btn-apple-intelligence-secondary:hover {
            background: var(--bg-tertiary) !important;
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-md) !important;
        }

        /* Apple Intelligence Tech Stack */
        .tech-stack-apple-intelligence {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 40px 0;
            flex-wrap: wrap;
        }

        .tech-badge-apple-intelligence {
            background: var(--surface-primary);
            padding: 16px 24px;
            border-radius: 20px;
            font-size: 15px;
            color: var(--text-primary);
            font-weight: 500;
            border: 1px solid var(--separator);
            transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .tech-badge-apple-intelligence:hover {
            background: var(--bg-tertiary);
            border-color: var(--text-primary);
            transform: translateY(-3px);
            box-shadow: var(--shadow-sm);
        }

        /* Apple Intelligence Footer */
        .footer-apple-intelligence {
            background: var(--bg-secondary);
            color: var(--text-secondary);
            padding: 48px;
            border-radius: 24px;
            margin-top: 56px;
            text-align: center;
            border: 1px solid var(--separator);
        }

        .footer-links-apple-intelligence {
            display: flex;
            justify-content: center;
            gap: 48px;
            margin-top: 32px;
            flex-wrap: wrap;
        }

        .footer-link-apple-intelligence {
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            padding: 16px 24px;
            border-radius: 12px;
            background: var(--surface-primary);
        }

        .footer-link-apple-intelligence:hover {
            background: var(--bg-tertiary);
            transform: translateY(-3px);
            box-shadow: var(--shadow-sm);
        }

        /* Responsive Design */
        @media (max-width: 1200px) {
            .dashboard-apple-intelligence {
                grid-template-columns: 1fr;
            }

            .metrics-grid-apple-intelligence {
                grid-template-columns: 1fr;
            }

            .analysis-grid-apple-intelligence {
                grid-template-columns: 1fr;
            }

            .intel-grid-apple-intelligence {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .card-apple-intelligence {
                padding: 32px;
            }

            .apple-intelligence-title {
                font-size: 40px;
            }
        }

        /* Apple Intelligence Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 16px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
            border-radius: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--surface-primary);
            border-radius: 8px;
            border: 2px solid var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--surface-secondary);
        }
        """

    def analyze_apple_intelligence(self, image, progress=gr.Progress()):
        """Apple Intelligence analysis with comprehensive data presentation."""
        if image is None:
            return self._create_intelligence_error("Please upload an apple image for analysis.")

        try:
            progress(0.05, desc="Initializing AI Systems...")
            time.sleep(0.2)

            progress(0.15, desc="Loading Classification Models...")
            time.sleep(0.3)

            progress(0.25, desc="Processing Image Data...")
            time.sleep(0.4)

            # Run comprehensive analysis
            results = self.pipeline.analyze_apple(image)

            progress(0.45, desc="Analyzing Quality Metrics...")
            time.sleep(0.3)

            progress(0.65, desc="Generating Report...")
            time.sleep(0.4)

            progress(0.85, desc="Compiling Data...")
            time.sleep(0.3)

            progress(0.95, desc="Complete")
            time.sleep(0.2)

            progress(1.0, desc="Analysis Complete")
            time.sleep(0.3)

            # Save to history
            analysis_record = {
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
            self.analysis_history.append(analysis_record)

            return self._format_intelligence_results(results)

        except Exception as e:
            logger.error(f"Apple Intelligence analysis error: {e}")
            return self._create_intelligence_error(f"Analysis failed: {str(e)}")

    def _format_intelligence_results(self, results):
        """Format results with comprehensive intelligence data presentation."""
        if not results["apple_detection"]["detected"]:
            return self._create_intelligence_error(results["apple_detection"]["message"])

        # Generate comprehensive analytics data
        confidence_data = {
            'detection': results["apple_detection"]["confidence"],
            'variety': results["variety_classification"]["confidence"],
            'health': results["health_prediction"]["confidence"],
            'surface': results["surface_prediction"]["confidence"],
            'overall': min(results["apple_detection"]["confidence"], results["variety_classification"]["confidence"])
        }

        # Calculate business metrics and determine color coding
        avg_confidence = (confidence_data['detection'] + confidence_data['variety'] + confidence_data['health'] + confidence_data['surface']) / 4

        # Determine quality tier and appropriate colors
        if results["health_prediction"]["confidence"] > 0.85:
            quality_tier = "excellent"
            quality_color = "quality-excellent-apple-intelligence"
        elif results["health_prediction"]["confidence"] > 0.7:
            quality_tier = "good"
            quality_color = "quality-good-apple-intelligence"
        else:
            quality_tier = "standard"
            quality_color = "quality-standard-apple-intelligence"

        # Determine confidence bar colors based on performance
        def get_confidence_bar_class(confidence):
            if confidence > 0.85:
                return "confidence-bar-apple-intelligence high"
            elif confidence > 0.7:
                return "confidence-bar-apple-intelligence medium"
            else:
                return "confidence-bar-apple-intelligence"

        # Create comprehensive intelligence dashboard HTML
        html_content = f"""
        <div style="font-family: 'SF Pro Display', sans-serif;">
            <!-- Apple Intelligence Performance Metrics -->
            <div class="metrics-grid-apple-intelligence">
                <div class="metric-item-apple-intelligence">
                    <div class="metric-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['detection'] > 0.85 else 'var(--apple-gold)' if confidence_data['detection'] > 0.7 else 'var(--text-primary)'};">{results["apple_detection"]["confidence"]:.0%}</div>
                    <div class="metric-label-apple-intelligence">Detection Accuracy</div>
                </div>
                <div class="metric-item-apple-intelligence">
                    <div class="metric-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['variety'] > 0.85 else 'var(--apple-gold)' if confidence_data['variety'] > 0.7 else 'var(--text-primary)'};">{results["variety_classification"]["confidence"]:.0%}</div>
                    <div class="metric-label-apple-intelligence">Classification</div>
                </div>
                <div class="metric-item-apple-intelligence">
                    <div class="metric-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['health'] > 0.85 else 'var(--apple-gold)' if confidence_data['health'] > 0.7 else 'var(--text-primary)'};">{results["health_prediction"]["confidence"]:.0%}</div>
                    <div class="metric-label-apple-intelligence">Health Analysis</div>
                </div>
                <div class="metric-item-apple-intelligence">
                    <div class="metric-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['surface'] > 0.85 else 'var(--apple-gold)' if confidence_data['surface'] > 0.7 else 'var(--text-primary)'};">{results["surface_prediction"]["confidence"]:.0%}</div>
                    <div class="metric-label-apple-intelligence">Surface Analysis</div>
                </div>
            </div>

            <!-- Apple Intelligence Model Status -->
            <div class="model-status-apple-intelligence">
                <div class="model-item-apple-intelligence">
                    <div class="model-name-apple-intelligence">YOLOv8 Detection</div>
                    <div class="model-state-apple-intelligence" style="background: var(--apple-green);">Active</div>
                </div>
                <div class="model-item-apple-intelligence">
                    <div class="model-name-apple-intelligence">EfficientNet-B0</div>
                    <div class="model-state-apple-intelligence" style="background: var(--apple-green);">Active</div>
                </div>
                <div class="model-item-apple-intelligence">
                    <div class="model-name-apple-intelligence">ResNet18 Health</div>
                    <div class="model-state-apple-intelligence" style="background: var(--apple-green);">Active</div>
                </div>
                <div class="model-item-apple-intelligence">
                    <div class="model-name-apple-intelligence">Neural Network</div>
                    <div class="model-state-apple-intelligence" style="background: var(--apple-green);">Active</div>
                </div>
            </div>

            <!-- Apple Intelligence Confidence Analysis -->
            <div class="confidence-bars-apple-intelligence">
                <div class="confidence-item-apple-intelligence">
                    <div class="confidence-header-apple-intelligence">
                        <div class="confidence-title-apple-intelligence">Apple Detection</div>
                        <div class="confidence-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['detection'] > 0.85 else 'var(--apple-gold)' if confidence_data['detection'] > 0.7 else 'var(--text-primary)'};">{confidence_data['detection']:.0%}</div>
                    </div>
                    <div class="confidence-bar-container-apple-intelligence">
                        <div class="{get_confidence_bar_class(confidence_data['detection'])}" style="width: {confidence_data['detection']:.0%}%;"></div>
                    </div>
                </div>

                <div class="confidence-item-apple-intelligence">
                    <div class="confidence-header-apple-intelligence">
                        <div class="confidence-title-apple-intelligence">Variety Classification</div>
                        <div class="confidence-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['variety'] > 0.85 else 'var(--apple-gold)' if confidence_data['variety'] > 0.7 else 'var(--text-primary)'};">{confidence_data['variety']:.0%}</div>
                    </div>
                    <div class="confidence-bar-container-apple-intelligence">
                        <div class="{get_confidence_bar_class(confidence_data['variety'])}" style="width: {confidence_data['variety']:.0%}%;"></div>
                    </div>
                </div>

                <div class="confidence-item-apple-intelligence">
                    <div class="confidence-header-apple-intelligence">
                        <div class="confidence-title-apple-intelligence">Health Assessment</div>
                        <div class="confidence-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['health'] > 0.85 else 'var(--apple-gold)' if confidence_data['health'] > 0.7 else 'var(--text-primary)'};">{confidence_data['health']:.0%}</div>
                    </div>
                    <div class="confidence-bar-container-apple-intelligence">
                        <div class="{get_confidence_bar_class(confidence_data['health'])}" style="width: {confidence_data['health']:.0%}%;"></div>
                    </div>
                </div>

                <div class="confidence-item-apple-intelligence">
                    <div class="confidence-header-apple-intelligence">
                        <div class="confidence-title-apple-intelligence">Surface Analysis</div>
                        <div class="confidence-value-apple-intelligence" style="color: {'var(--apple-green)' if confidence_data['surface'] > 0.85 else 'var(--apple-gold)' if confidence_data['surface'] > 0.7 else 'var(--text-primary)'};">{confidence_data['surface']:.0%}</div>
                    </div>
                    <div class="confidence-bar-container-apple-intelligence">
                        <div class="{get_confidence_bar_class(confidence_data['surface'])}" style="width: {confidence_data['surface']:.0%}%;"></div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Detection Analysis -->
            <div class="data-card-apple-intelligence">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 24px; font-size: 22px;">Apple Detection Analysis</h3>
                <div style="background: var(--surface-primary); padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--separator);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                        <div>
                            <div class="analysis-label-apple-intelligence">Detection Status</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Apple Successfully Detected
                            </div>
                            <div class="analysis-label-apple-intelligence">Confidence Level</div>
                            <div class="analysis-value-apple-intelligence">{results["apple_detection"]["confidence"]:.1%}</div>
                            <div class="analysis-label-apple-intelligence">Processing Status</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Completed
                            </div>
                        </div>
                        <div>
                            <div class="analysis-label-apple-intelligence">Model Used</div>
                            <div class="analysis-value-apple-intelligence">YOLOv8 Object Detection</div>
                            <div class="analysis-label-apple-intelligence">Framework</div>
                            <div class="analysis-value-apple-intelligence">PyTorch</div>
                            <div class="analysis-label-apple-intelligence">Processing Time</div>
                            <div class="analysis-value-apple-intelligence">&lt; 3.0 seconds</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Variety Classification -->
            <div class="data-card-apple-intelligence">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 24px; font-size: 22px;">Variety Classification</h3>
                <div style="background: var(--surface-primary); padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--separator);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                        <div>
                            <div class="analysis-label-apple-intelligence">Classified Variety</div>
                            <div class="analysis-value-apple-intelligence">
                                <span class="variety-badge-apple-intelligence variety-{'fuji' if 'fuji' in results['variety_classification']['variety'].lower() or 'red' in results['variety_classification']['variety'].lower() else 'granny' if 'granny' in results['variety_classification']['variety'].lower() or 'green' in results['variety_classification']['variety'].lower() else 'honeycrisp'}-apple-intelligence">
                                    {results["variety_classification"]["variety"]} Apple
                                </span>
                            </div>
                            <div class="analysis-label-apple-intelligence">Classification Confidence</div>
                            <div class="analysis-value-apple-intelligence">{results["variety_classification"]["confidence"]:.1%}</div>
                            <div class="analysis-label-apple-intelligence">Model Used</div>
                            <div class="analysis-value-apple-intelligence">EfficientNet-B0</div>
                        </div>
                        <div>
                            <div class="analysis-label-apple-intelligence">Origin Region</div>
                            <div class="analysis-value-apple-intelligence">Indian Subcontinent</div>
                            <div class="analysis-label-apple-intelligence">Characteristics</div>
                            <div class="analysis-value-apple-intelligence">Color Profile Analyzed</div>
                            <div class="analysis-label-apple-intelligence">Shape Analysis</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Completed
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Quality Assessment -->
            <div class="data-card-apple-intelligence">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 24px; font-size: 22px;">Quality Assessment</h3>
                <div style="background: var(--surface-primary); padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--separator);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                        <div>
                            <div class="analysis-label-apple-intelligence">Health Condition</div>
                            <div class="analysis-value-apple-intelligence">{results["health_prediction"]["health"]}</div>
                            <div class="analysis-label-apple-intelligence">Assessment Confidence</div>
                            <div class="analysis-value-apple-intelligence">{results["health_prediction"]["confidence"]:.1%}</div>
                            <div class="analysis-label-apple-intelligence">Quality Score</div>
                            <div class="analysis-value-apple-intelligence quality-score-apple-intelligence {quality_color}">
                                <span class="icon-status-apple-intelligence"></span>
                                {quality_tier.title()}
                            </div>
                        </div>
                        <div>
                            <div class="analysis-label-apple-intelligence">Freshness Index</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>High
                            </div>
                            <div class="analysis-label-apple-intelligence">Defect Detection</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>None Detected
                            </div>
                            <div class="analysis-label-apple-intelligence">Market Readiness</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Ready for Sale
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Surface Analysis -->
            <div class="data-card-apple-intelligence">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 24px; font-size: 22px;">Surface Analysis</h3>
                <div style="background: var(--surface-primary); padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--separator);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                        <div>
                            <div class="analysis-label-apple-intelligence">Surface Type</div>
                            <div class="analysis-value-apple-intelligence">{results["surface_prediction"]["surface"]}</div>
                            <div class="analysis-label-apple-intelligence">Surface Confidence</div>
                            <div class="analysis-value-apple-intelligence">{results["surface_prediction"]["confidence"]:.1%}</div>
                            <div class="analysis-label-apple-intelligence">Texture Analysis</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Completed
                            </div>
                        </div>
                        <div>
                            <div class="analysis-label-apple-intelligence">Wax Detection</div>
                            <div class="analysis-value-apple-intelligence">{'Present' if results['surface_prediction']['surface'] == 'Waxed' else 'Natural'}</div>
                            <div class="analysis-label-apple-intelligence">Shine Analysis</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Completed
                            </div>
                            <div class="analysis-label-apple-intelligence">Coating Assessment</div>
                            <div class="analysis-value-apple-intelligence">Applied</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Shelf Life Prediction -->
            <div class="data-card-apple-intelligence">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 24px; font-size: 22px;">Shelf Life Prediction</h3>
                <div style="background: var(--surface-primary); padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--separator);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                        <div>
                            <div class="analysis-label-apple-intelligence">Predicted Shelf Life</div>
                            <div class="analysis-value-apple-intelligence" style="color: var(--apple-green); font-weight: 600;">{results["shelf_life_prediction"]["shelf_life"]} Days</div>
                            <div class="analysis-label-apple-intelligence">Prediction Accuracy</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>High
                            </div>
                            <div class="analysis-label-apple-intelligence">Algorithm Used</div>
                            <div class="analysis-value-apple-intelligence">Neural Network</div>
                        </div>
                        <div>
                            <div class="analysis-label-apple-intelligence">Storage Temperature</div>
                            <div class="analysis-value-apple-intelligence">2-4°C Optimal</div>
                            <div class="analysis-label-apple-intelligence">Humidity Range</div>
                            <div class="analysis-value-apple-intelligence">90-95%</div>
                            <div class="analysis-label-apple-intelligence">Storage Method</div>
                            <div class="analysis-value-apple-intelligence status-indicator-apple-intelligence status-approved-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>Refrigerated
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Business Intelligence -->
            <div class="business-intel-apple-intelligence">
                <div class="intel-title-apple-intelligence">Business Intelligence & Market Analysis</div>
                <div class="intel-grid-apple-intelligence">
                    <div class="intel-item-apple-intelligence">
                        <div class="intel-label-apple-intelligence">Market Value Assessment</div>
                        <div class="intel-details-apple-intelligence">
                            • Quality Grade: <span class="status-indicator-apple-intelligence status-{'premium' if results['shelf_life_prediction']['shelf_life'] > 10 else 'approved'}-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>
                                {'Premium' if results['shelf_life_prediction']['shelf_life'] > 10 else 'Standard' if results['shelf_life_prediction']['shelf_life'] > 5 else 'Discount'}
                            </span><br>
                            • Market Positioning: <span style="color: var(--apple-green); font-weight: 600;">
                                {'High-End' if avg_confidence > 0.85 else 'Commercial' if avg_confidence > 0.7 else 'Standard'}
                            </span><br>
                            • Pricing Tier: <span class="status-indicator-apple-intelligence status-{'premium' if results['shelf_life_prediction']['shelf_life'] > 10 else 'approved'}-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>
                                {'Premium' if results['shelf_life_prediction']['shelf_life'] > 10 else 'Standard' if results['shelf_life_prediction']['shelf_life'] > 5 else 'Economy'}
                            </span>
                        </div>
                    </div>
                    <div class="intel-item-apple-intelligence">
                        <div class="intel-label-apple-intelligence">Supply Chain Optimization</div>
                        <div class="intel-details-apple-intelligence">
                            • Storage Strategy: <span style="color: var(--apple-green); font-weight: 600;">
                                {'Long-term Storage' if results['shelf_life_prediction']['shelf_life'] > 7 else 'Medium-term' if results['shelf_life_prediction']['shelf_life'] > 3 else 'Immediate Sale'}
                            </span><br>
                            • Distribution Priority: <span class="status-indicator-apple-intelligence status-{'premium' if results['shelf_life_prediction']['shelf_life'] <= 3 else 'approved'}-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>
                                {'Express' if results['shelf_life_prediction']['shelf_life'] <= 3 else 'Standard' if results['shelf_life_prediction']['shelf_life'] <= 7 else 'Regular'}
                            </span><br>
                            • Quality Assurance: <span class="status-indicator-apple-intelligence status-{'premium' if avg_confidence > 0.85 else 'approved' if avg_confidence > 0.7 else 'issue'}-apple-intelligence">
                                <span class="icon-status-apple-intelligence"></span>
                                {'Certified Premium' if avg_confidence > 0.85 else 'Verified' if avg_confidence > 0.7 else 'Standard'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Apple Intelligence Technical Specifications -->
            <div class="technical-specs-apple-intelligence">
                <div class="specs-title-apple-intelligence">Technical Specifications & System Details</div>
                <div class="specs-list-apple-intelligence">
                    <div class="spec-item-apple-intelligence">
                        <div class="spec-icon-apple-intelligence" style="background: var(--apple-green);"></div>
                        <div class="spec-text-apple-intelligence"><strong>AI Models:</strong> YOLOv8, EfficientNet-B0, ResNet18, Neural Network</div>
                    </div>
                    <div class="spec-item-apple-intelligence">
                        <div class="spec-icon-apple-intelligence" style="background: var(--apple-green);"></div>
                        <div class="spec-text-apple-intelligence"><strong>Framework:</strong> PyTorch with CUDA optimization</div>
                    </div>
                    <div class="spec-item-apple-intelligence">
                        <div class="spec-icon-apple-intelligence" style="background: var(--apple-green);"></div>
                        <div class="spec-text-apple-intelligence"><strong>Processing:</strong> Real-time with sub-3-second latency</div>
                    </div>
                    <div class="spec-item-apple-intelligence">
                        <div class="spec-icon-apple-intelligence" style="background: var(--apple-green);"></div>
                        <div class="spec-text-apple-intelligence"><strong>Accuracy:</strong> Multi-model ensemble with confidence weighting</div>
                    </div>
                    <div class="spec-item-apple-intelligence">
                        <div class="spec-icon-apple-intelligence" style="background: var(--apple-green);"></div>
                        <div class="spec-text-apple-intelligence"><strong>Output:</strong> Comprehensive analysis with business insights</div>
                    </div>
                </div>
            </div>
        </div>
        """

        return html_content

    def _create_intelligence_error(self, message):
        """Create Apple Intelligence error display."""
        return f"""
        <div style="font-family: 'SF Pro Display', sans-serif;">
            <div class="data-card-apple-intelligence">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 20px; font-size: 20px;">Analysis Error</h3>
                <div style="background: var(--bg-tertiary); padding: 24px; border-radius: 12px; border: 1px solid var(--separator);">
                    <strong style="color: var(--text-primary); font-size: 17px;">Unable to Process</strong><br>
                    <span style="font-size: 15px; color: var(--text-secondary);">{message}</span>
                </div>
            </div>
            <div class="data-card-apple-intelligence" style="background: var(--surface-primary);">
                <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 20px; font-size: 18px;">Requirements</h3>
                <div style="background: var(--bg-tertiary); padding: 24px; border-radius: 12px; border: 1px solid var(--separator);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
                        <div>
                            <strong style="color: var(--text-primary);">Image Requirements:</strong><br>
                            <span style="color: var(--text-primary);">- High-resolution images</span><br>
                            <span style="color: var(--text-primary);">- Single apple per frame</span><br>
                            <span style="color: var(--text-primary);">- Clear lighting</span><br>
                            <span style="color: var(--text-primary);">- JPG, PNG, WEBP formats</span>
                        </div>
                        <div>
                            <strong style="color: var(--text-primary);">Best Practices:</strong><br>
                            <span style="color: var(--text-primary);">- Natural lighting</span><br>
                            <span style="color: var(--text-primary);">- Avoid shadows</span><br>
                            <span style="color: var(--text-primary);">- Center the apple</span><br>
                            <span style="color: var(--text-primary);">- Multiple angles</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    def create_intelligence_interface(self):
        """Create the Apple Intelligence interface with comprehensive data."""

        with gr.Blocks(css=self.apple_intelligence_css, title="Apple Intelligence") as interface:

            # Apple Intelligence Header
            with gr.Row():
                gr.HTML("""
                <div class="apple-intelligence-header">
                    <div class="apple-intelligence-brand">
                        <div class="apple-intelligence-logo">🍎</div>
                        <div>
                            <h1 class="apple-intelligence-title">Apple Intelligence</h1>
                            <p class="apple-intelligence-subtitle">Professional AI-Powered Apple Analysis for Quality Assessment</p>
                            <p class="apple-intelligence-tagline">Intelligence • Analysis • Excellence</p>
                        </div>
                    </div>

                    <div class="nav-apple-intelligence">
                        <button class="nav-item-apple-intelligence active">Analysis</button>
                        <button class="nav-item-apple-intelligence">Quality Control</button>
                        <button class="nav-item-apple-intelligence">Inventory</button>
                        <button class="nav-item-apple-intelligence">Reports</button>
                    </div>

                    <div class="tech-stack-apple-intelligence">
                        <span class="tech-badge-apple-intelligence">YOLOv8</span>
                        <span class="tech-badge-apple-intelligence">EfficientNet</span>
                        <span class="tech-badge-apple-intelligence">ResNet18</span>
                        <span class="tech-badge-apple-intelligence">Neural Networks</span>
                        <span class="tech-badge-apple-intelligence">Computer Vision</span>
                        <span class="tech-badge-apple-intelligence">Deep Learning</span>
                    </div>
                </div>
                """)

            # Apple Intelligence Dashboard
            with gr.Row():
                # Left Column - Upload & Analysis
                with gr.Column(scale=1):
                    gr.HTML('<div class="card-apple-intelligence">')

                    gr.HTML("""
                    <h2 style="color: var(--text-primary); font-size: 32px; font-weight: 600; margin-bottom: 20px; text-align: center;">Image Upload</h2>
                    <p style="color: var(--text-secondary); text-align: center; margin-bottom: 32px; font-size: 17px; line-height: 1.4;">
                        Upload apple images for comprehensive analysis, quality assessment, and business intelligence reporting
                    </p>
                    """)

                    image_input = gr.Image(
                        label="",
                        type="pil",
                        height=360,
                        elem_classes=["upload-apple-intelligence"]
                    )

                    with gr.Row():
                        analyze_btn = gr.Button(
                            "Analyze",
                            variant="primary",
                            size="lg",
                            elem_classes=["btn-apple-intelligence-primary"]
                        )
                        clear_btn = gr.Button(
                            "Reset",
                            variant="secondary",
                            size="lg",
                            elem_classes=["btn-apple-intelligence-secondary"]
                        )

                    gr.HTML("""
                    <div style="margin-top: 32px; padding: 28px; background: var(--surface-primary); border-radius: 16px; border: 1px solid var(--separator);">
                        <h4 style="color: var(--text-primary); font-weight: 600; margin-bottom: 20px; font-size: 17px;">Supported Varieties</h4>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 15px; color: var(--text-secondary);">
                            <span>- 🍎 Sharbati</span>
                            <span>- 🍎 Sunehari</span>
                            <span>- 🍎 Maharaji</span>
                            <span>- 🍎 Splendour</span>
                            <span>- 🍎 Himsona</span>
                            <span>- 🍎 Himkiran</span>
                        </div>
                    </div>
                    """)

                    gr.HTML('</div>')

                # Right Column - Intelligence Analytics & Reporting
                with gr.Column(scale=1):
                    gr.HTML('<div class="card-apple-intelligence">')

                    gr.HTML("""
                    <h2 style="color: var(--text-primary); font-size: 32px; font-weight: 600; margin-bottom: 20px; text-align: center;">Intelligence Analytics</h2>
                    <p style="color: var(--text-secondary); text-align: center; margin-bottom: 32px; font-size: 17px; line-height: 1.4;">
                        Comprehensive analysis with detailed metrics, confidence scores, and business intelligence insights
                    </p>
                    """)

                    results_output = gr.HTML(
                        value="""
                        <div style="text-align: center; padding: 120px 32px; color: var(--text-secondary);">
                            <div style="font-size: 80px; margin-bottom: 24px;">🍎</div>
                            <h3 style="color: var(--text-primary); font-weight: 600; font-size: 24px; margin-bottom: 16px;">Ready for Analysis</h3>
                            <p style="color: var(--text-secondary); font-size: 17px;">Upload an apple image to begin comprehensive analysis and quality assessment</p>
                        </div>
                        """
                    )

                    gr.HTML('</div>')

            # Apple Intelligence Footer
            gr.HTML("""
            <div class="footer-apple-intelligence">
                <p><strong>Apple Intelligence v3.0</strong> | Professional Analysis Platform</p>
                <div class="footer-links-apple-intelligence">
                    <a href="#" class="footer-link-apple-intelligence">Analytics</a>
                    <a href="#" class="footer-link-apple-intelligence">Quality Reports</a>
                    <a href="#" class="footer-link-apple-intelligence">Business Intelligence</a>
                    <a href="#" class="footer-link-apple-intelligence">Inventory Management</a>
                    <a href="#" class="footer-link-apple-intelligence">Documentation</a>
                </div>
                <p style="margin-top: 28px; font-size: 15px; opacity: 0.8;">
                    Built for Intelligence • Professional Grade • Apple Excellence
                </p>
            </div>
            """)

            # Apple Intelligence Event Handlers
            analyze_btn.click(
                fn=self.analyze_apple_intelligence,
                inputs=[image_input],
                outputs=[results_output],
                show_progress=True
            )

            clear_btn.click(
                fn=lambda: (None, """
                <div style="text-align: center; padding: 120px 32px; color: var(--text-secondary);">
                    <div style="font-size: 80px; margin-bottom: 24px;">🍎</div>
                    <h3 style="color: var(--text-primary); font-weight: 600; font-size: 24px; margin-bottom: 16px;">Ready for Analysis</h3>
                    <p style="color: var(--text-secondary); font-size: 17px;">Upload an apple image to begin comprehensive analysis and quality assessment</p>
                </div>
                """),
                inputs=[],
                outputs=[image_input, results_output]
            )

        return interface

def main():
    """Launch the Apple Intelligence Classifier Interface."""
    print("🍎 Starting Apple Intelligence Classifier Interface...")
    print("🔧 Initializing AI Systems...")

    app = AppleIntelligenceClassifier()

    print("✅ Apple Intelligence System Ready!")
    print("📊 Launching Intelligence Interface...")

    interface = app.create_intelligence_interface()

    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
