const form = document.getElementById('predictionForm');
const button = document.getElementById('predictBtn');
const placeholder = document.querySelector('.result-placeholder');
const content = document.querySelector('.result-content');
const title = document.getElementById('resultTitle');
const risk = document.getElementById('riskLevel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const message = document.getElementById('resultMessage');
const initial = document.getElementById('resultInitial');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.querySelector('span:first-child').textContent = 'Analyzing...';

  const data = Object.fromEntries(new FormData(form).entries());
  for (const key of Object.keys(data)) data[key] = Number(data[key]);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.error || 'Prediction failed');

    placeholder.classList.add('hidden');
    content.classList.remove('hidden');
    title.textContent = result.result || 'Unknown';
    risk.textContent = result.risk_level || 'Unknown';
    message.textContent = result.message || 'Prediction generated successfully.';

    const confidence = Number(result.confidence || 0);
    confidenceValue.textContent = `${confidence.toFixed(2)}%`;
    confidenceBar.style.width = `${Math.min(Math.max(confidence, 0), 100)}%`;
    initial.textContent = (result.result || '?').charAt(0);

    if ((result.result || '').toLowerCase().includes('diabetes')) {
      risk.style.background = '#fff0f1';
      risk.style.color = '#c33a4b';
    } else if ((result.result || '').toLowerCase().includes('pre')) {
      risk.style.background = '#fff7df';
      risk.style.color = '#a36a00';
    } else {
      risk.style.background = '#eaf8f1';
      risk.style.color = '#178957';
    }
  } catch (error) {
    placeholder.classList.remove('hidden');
    content.classList.add('hidden');
    placeholder.querySelector('h3').textContent = 'Something went wrong';
    placeholder.querySelector('p').textContent = error.message;
  } finally {
    button.disabled = false;
    button.querySelector('span:first-child').textContent = 'Run Screening';
  }
});