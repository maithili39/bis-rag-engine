document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const queryInput = document.getElementById('product-query');
    const btnText = searchBtn.querySelector('.btn-text');
    const spinner = searchBtn.querySelector('.spinner');
    
    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');
    
    const resultsMeta = document.getElementById('results-meta');
    const standardsList = document.getElementById('standards-list');
    const rationaleText = document.getElementById('rationale-text');
    const errorText = document.getElementById('error-text');

    async function performSearch() {
        const query = queryInput.value.trim();
        if (!query) {
            showError("Please enter a product, material, or process description.");
            return;
        }

        // Reset UI state
        errorContainer.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        standardsList.innerHTML = '';
        
        // Loading state
        searchBtn.disabled = true;
        btnText.textContent = "Searching...";
        spinner.classList.remove('hidden');

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query, top_k: 5 })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            showError("Failed to fetch standards. Please ensure the backend is running and vectorstore is built.");
            console.error(err);
        } finally {
            // Restore button state
            searchBtn.disabled = false;
            btnText.textContent = "Search Standards";
            spinner.classList.add('hidden');
        }
    }

    function displayResults(data) {
        const { recommended_standards, rationale, latency_seconds } = data;
        
        resultsMeta.textContent = `${recommended_standards.length} standards found • ${latency_seconds.toFixed(2)}s latency`;
        
        recommended_standards.forEach((code, index) => {
            const isTopMatch = index === 0;
            const card = document.createElement('div');
            card.className = `standard-card ${isTopMatch ? 'top-match' : ''}`;
            
            card.innerHTML = `
                <div class="rank-badge">#${String(index + 1).padStart(2, '0')}</div>
                <div class="standard-code">${code}</div>
                ${isTopMatch ? '<div class="top-match-tag">Top Match</div>' : ''}
            `;
            standardsList.appendChild(card);
        });

        rationaleText.textContent = rationale;
        resultsContainer.classList.remove('hidden');
    }

    function showError(message) {
        errorText.textContent = message;
        errorContainer.classList.remove('hidden');
    }

    searchBtn.addEventListener('click', performSearch);

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            performSearch();
        }
    });
});
