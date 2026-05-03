import os
import sys
import time
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from src.rag_pipeline import BISRAGPipeline
from src.config import CHROMA_PERSIST_DIR, DATASET_PDF

app = Flask(__name__)
CORS(app)

# Global pipeline instance
pipeline = None

# Simple inline HTML for the UI (so we don't need a separate templates folder)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BIS Standards AI Recommender</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
        .glass-panel {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .standard-card {
            transition: all 0.2s ease;
            border-left: 4px solid transparent;
        }
        .standard-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3b82f6;
        }
        .loading-spinner {
            border: 3px solid rgba(59, 130, 246, 0.2);
            border-radius: 50%;
            border-top: 3px solid #3b82f6;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="min-h-screen text-slate-800">
    <!-- Navbar -->
    <nav class="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16 items-center">
                <div class="flex items-center">
                    <div class="flex-shrink-0 flex items-center gap-3">
                        <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold">B</div>
                        <span class="font-bold text-xl tracking-tight">BIS Recommender AI</span>
                    </div>
                </div>
                <div class="flex items-center gap-4 text-sm font-medium text-slate-500">
                    <span class="bg-green-100 text-green-700 px-3 py-1 rounded-full text-xs">SP 21 Dataset Loaded</span>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div class="text-center mb-10">
            <h1 class="text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl mb-4">
                Find Your Standard Instantly
            </h1>
            <p class="max-w-2xl mx-auto text-lg text-slate-500">
                Describe your manufacturing product or material, and our AI will retrieve the exact Bureau of Indian Standards (BIS) regulations you need to comply with.
            </p>
        </div>

        <div class="glass-panel p-6 sm:p-8 mb-10">
            <form id="search-form" class="relative">
                <div class="flex flex-col sm:flex-row gap-4">
                    <div class="relative flex-grow">
                        <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                            <svg class="h-5 w-5 text-slate-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <input 
                            type="text" 
                            id="query-input" 
                            class="block w-full pl-11 pr-4 py-4 bg-white border border-slate-300 rounded-xl text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow text-lg" 
                            placeholder="e.g., We manufacture 33 Grade Ordinary Portland Cement..." 
                            required
                        >
                    </div>
                    <button 
                        type="submit" 
                        class="inline-flex justify-center items-center px-8 py-4 border border-transparent text-base font-medium rounded-xl text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors shadow-sm"
                    >
                        Analyze
                    </button>
                </div>
            </form>
            
            <div class="mt-4 flex gap-2 text-sm text-slate-500 overflow-x-auto pb-2">
                <span class="whitespace-nowrap font-medium text-slate-700">Try asking:</span>
                <button class="suggestion-btn whitespace-nowrap bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-md transition-colors" onclick="setQuery('coarse and fine aggregates for structural concrete')">Aggregates</button>
                <button class="suggestion-btn whitespace-nowrap bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-md transition-colors" onclick="setQuery('precast concrete pipes with reinforcement for water mains')">Concrete Pipes</button>
                <button class="suggestion-btn whitespace-nowrap bg-slate-100 hover:bg-slate-200 px-3 py-1 rounded-md transition-colors" onclick="setQuery('Portland slag cement manufacture and testing')">Slag Cement</button>
            </div>
        </div>

        <!-- Loading State -->
        <div id="loading" class="hidden flex flex-col items-center justify-center py-12">
            <div class="loading-spinner mb-4"></div>
            <p class="text-slate-500 font-medium animate-pulse">Scanning BIS SP 21 Knowledge Base...</p>
        </div>

        <!-- Results Section -->
        <div id="results-container" class="hidden">
            <div class="flex justify-between items-end mb-6">
                <h2 class="text-2xl font-bold text-slate-900">Recommended Standards</h2>
                <div class="text-sm text-slate-500 bg-white px-3 py-1 rounded-md border border-slate-200 shadow-sm">
                    Latency: <span id="latency-val" class="font-mono font-bold text-blue-600">0.00</span>s
                </div>
            </div>
            
            <div id="standards-list" class="space-y-4">
                <!-- Results populated via JS -->
            </div>
        </div>
        
        <!-- Empty State -->
        <div id="empty-state" class="hidden text-center py-16 bg-white rounded-2xl border border-slate-200 border-dashed">
            <svg class="mx-auto h-12 w-12 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
            </svg>
            <h3 class="mt-2 text-sm font-medium text-slate-900">No standards found</h3>
            <p class="mt-1 text-sm text-slate-500">Try adjusting your description or using different keywords.</p>
        </div>
    </main>

    <script>
        function setQuery(text) {
            document.getElementById('query-input').value = text;
            document.getElementById('search-form').dispatchEvent(new Event('submit'));
        }

        document.getElementById('search-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('query-input').value.trim();
            if (!query) return;
            
            // UI State updates
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results-container').classList.add('hidden');
            document.getElementById('empty-state').classList.add('hidden');
            
            try {
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                document.getElementById('loading').classList.add('hidden');
                
                if (data.details && data.details.length > 0) {
                    document.getElementById('results-container').classList.remove('hidden');
                    document.getElementById('latency-val').textContent = data.latency_seconds.toFixed(2);
                    
                    const listContainer = document.getElementById('standards-list');
                    listContainer.innerHTML = '';
                    
                    data.details.forEach((item, index) => {
                        const isTopResult = index === 0;
                        const card = document.createElement('div');
                        
                        // Extract just the scope/content to show a preview
                        let snippet = item.snippet || '';
                        if (snippet.length > 250) snippet = snippet.substring(0, 250) + '...';
                        
                        card.className = `standard-card bg-white rounded-xl p-6 border border-slate-200 shadow-sm ${isTopResult ? 'ring-1 ring-blue-500 border-transparent' : ''}`;
                        card.innerHTML = `
                            <div class="flex items-start justify-between gap-4">
                                <div>
                                    <div class="flex items-center gap-3 mb-2">
                                        <h3 class="text-xl font-bold text-slate-900">${item.code}</h3>
                                        ${isTopResult ? '<span class="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded border border-blue-200">Best Match</span>' : ''}
                                    </div>
                                    <h4 class="text-md font-semibold text-slate-700 mb-3">${item.title}</h4>
                                    <p class="text-slate-600 text-sm leading-relaxed">${snippet}</p>
                                </div>
                                <div class="flex-shrink-0 flex flex-col items-end">
                                    <div class="flex items-center gap-1 text-sm ${item.score > 0.6 ? 'text-green-600' : 'text-amber-600'} font-medium bg-slate-50 px-3 py-1 rounded-md border border-slate-100">
                                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>
                                        ${(item.score * 100).toFixed(0)}% Match
                                    </div>
                                </div>
                            </div>
                        `;
                        listContainer.appendChild(card);
                    });
                } else {
                    document.getElementById('empty-state').classList.remove('hidden');
                }
                
            } catch (error) {
                console.error("Error:", error);
                document.getElementById('loading').classList.add('hidden');
                alert("Failed to connect to the recommendation engine. Please ensure the server is running.");
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    global pipeline
    if not pipeline:
        return jsonify({"error": "Pipeline not initialized"}), 500
        
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
        
    query = data['query']
    
    try:
        # Use detailed inference for UI
        results = pipeline.process_query_detailed(query, top_k=5)
        return jsonify(results)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

def init_app():
    global pipeline
    print("Initializing RAG Engine for Web UI...")
    
    pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
    
    if not pipeline.load_existing_vectorstore():
        print("Vectorstore not found. Attempting to build from PDF...")
        if os.path.exists(DATASET_PDF):
            pipeline.initialize_from_pdf(DATASET_PDF)
        else:
            print(f"ERROR: Dataset PDF not found at {DATASET_PDF}")
            print("Please run: python build_vectorstore.py")
            sys.exit(1)
            
    print("Pipeline ready!")

if __name__ == '__main__':
    init_app()
    print("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
