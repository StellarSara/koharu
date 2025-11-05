/**
 * TypeScript/React Integration Example for PaddleOCR
 * 
 * Add these functions to your store or component to use PaddleOCR
 */

import { invoke } from '@tauri-apps/api/core'

// Type definitions
interface Document {
  id: string
  path: string
  name: string
  image: any
  width: number
  height: number
  textBlocks: TextBlock[]
  segment?: any
  inpainted?: any
}

interface TextBlock {
  x: number
  y: number
  width: number
  height: number
  confidence: number
  text?: string
  translation?: string
}

// Store actions (add to your existing store)
export const paddleOcrActions = {
  /**
   * Use PaddleOCR for recognition on already detected regions
   * Call this after detect() instead of ocr()
   */
  async ocrPaddle(index: number): Promise<Document> {
    return await invoke('ocr_paddle', { index })
  },

  /**
   * Full pipeline: detect and recognize using PaddleOCR
   * One-step process for Chinese text
   */
  async detectAndOcrPaddle(index: number): Promise<Document> {
    return await invoke('detect_and_ocr_paddle', { index })
  },
}

// Example Component Usage
export function OcrButtons() {
  const currentIndex = 0 // Get from your state

  const handleJapaneseOcr = async () => {
    // For Japanese manga (existing workflow)
    await invoke('detect', { 
      index: currentIndex,
      confThreshold: 0.5,
      nmsThreshold: 0.4 
    })
    await invoke('ocr', { index: currentIndex })
  }

  const handleChineseOcrWithComicDetector = async () => {
    // Use comic-text-detector for detection, PaddleOCR for recognition
    await invoke('detect', { 
      index: currentIndex,
      confThreshold: 0.5,
      nmsThreshold: 0.4 
    })
    await invoke('ocr_paddle', { index: currentIndex })
  }

  const handleChineseOcrFull = async () => {
    // Use PaddleOCR for both detection and recognition
    await invoke('detect_and_ocr_paddle', { index: currentIndex })
  }

  return (
    <div className="ocr-controls">
      <button onClick={handleJapaneseOcr}>
        Japanese OCR
      </button>
      <button onClick={handleChineseOcrWithComicDetector}>
        Chinese OCR (Hybrid)
      </button>
      <button onClick={handleChineseOcrFull}>
        Chinese OCR (Full)
      </button>
    </div>
  )
}

// Example: Language selector
export function LanguageSelector({ onChange }: { onChange: (lang: string) => void }) {
  return (
    <select onChange={(e) => onChange(e.target.value)}>
      <option value="japanese">Japanese (Manga OCR)</option>
      <option value="chinese-hybrid">Chinese (Comic Detector + PaddleOCR)</option>
      <option value="chinese-full">Chinese (PaddleOCR Full)</option>
    </select>
  )
}

// Example: Integrated workflow with language selection
export async function performOcr(
  language: 'japanese' | 'chinese-hybrid' | 'chinese-full',
  index: number,
  confThreshold: number = 0.5,
  nmsThreshold: number = 0.4
): Promise<Document> {
  switch (language) {
    case 'japanese':
      await invoke('detect', { index, confThreshold, nmsThreshold })
      return await invoke('ocr', { index })
    
    case 'chinese-hybrid':
      await invoke('detect', { index, confThreshold, nmsThreshold })
      return await invoke('ocr_paddle', { index })
    
    case 'chinese-full':
      return await invoke('detect_and_ocr_paddle', { index })
    
    default:
      throw new Error(`Unknown language: ${language}`)
  }
}

// Example: Update existing store (if using Zustand)
/*
interface AppStore {
  // ... existing state
  selectedLanguage: 'japanese' | 'chinese-hybrid' | 'chinese-full'
  
  setLanguage: (lang: 'japanese' | 'chinese-hybrid' | 'chinese-full') => void
  ocr: () => Promise<void>
}

export const useAppStore = create<AppStore>((set, get) => ({
  // ... existing state
  selectedLanguage: 'japanese',
  
  setLanguage: (lang) => set({ selectedLanguage: lang }),
  
  ocr: async () => {
    const { selectedLanguage, activeIndex, detect } = get()
    
    if (selectedLanguage === 'japanese') {
      const doc = await invoke('ocr', { index: activeIndex })
      // update state...
    } else if (selectedLanguage === 'chinese-hybrid') {
      const doc = await invoke('ocr_paddle', { index: activeIndex })
      // update state...
    } else if (selectedLanguage === 'chinese-full') {
      const doc = await invoke('detect_and_ocr_paddle', { index: activeIndex })
      // update state...
    }
  }
}))
*/
