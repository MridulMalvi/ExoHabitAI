document.addEventListener('DOMContentLoaded', () => {
    const dashboardGrid = document.getElementById('dashboardGrid');
    const earthDefaults = {
        P_RADIUS: 1,
        P_MASS: 1,
        P_GRAVITY: 1,
        P_PERIOD: 365.25,
        P_TEMP_EQUIL: 288,
        S_MASS: 1,
        S_RADIUS: 1,
        S_TEMPERATURE: 5778,
        S_LUMINOSITY: 1
    };

    function applyDefaultsToForm(values) {
        Object.entries(values).forEach(([key, val]) => {
            const el = document.getElementById(key);
            if (el) el.value = val;
        });
    }

    function renderEarthBaseline() {
        applyDefaultsToForm(earthDefaults);
        // Keep result hidden to avoid implying a server prediction; show charts with baseline probability.
        document.getElementById('result').classList.add('hidden');
        updateDashboard(earthDefaults, 100);
    }

    // -- PREDICT LOGIC --
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const btn = document.getElementById('predictBtn');
        const resultDiv = document.getElementById('result');
        const title = document.getElementById('resultTitle');
        const probText = document.getElementById('resultProb');
        const meter = document.getElementById('probMeter');

        btn.textContent = "Calculating Orbit...";
        btn.disabled = true;

        const data = {
            P_RADIUS: parseFloat(document.getElementById('P_RADIUS').value),
            P_MASS: parseFloat(document.getElementById('P_MASS').value),
            P_GRAVITY: parseFloat(document.getElementById('P_GRAVITY').value),
            P_PERIOD: parseFloat(document.getElementById('P_PERIOD').value),
            P_TEMP_EQUIL: parseFloat(document.getElementById('P_TEMP_EQUIL').value),
            S_MASS: parseFloat(document.getElementById('S_MASS').value),
            S_RADIUS: parseFloat(document.getElementById('S_RADIUS').value),
            S_TEMPERATURE: parseFloat(document.getElementById('S_TEMPERATURE').value),
            S_LUMINOSITY: parseFloat(document.getElementById('S_LUMINOSITY').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            resultDiv.classList.remove('hidden');
            title.textContent = result.label;

            const percentage = Math.max(0, Math.min(100, Number(result.probability) || 0));
            probText.innerHTML = `Habitability Probability: <strong>${percentage}%</strong>`;
            meter.style.width = `${percentage}%`;

            if (result.prediction === 1) {
                resultDiv.className = "result-box good";
                meter.style.background = "#4bff64";
            } else {
                resultDiv.className = "result-box bad";
                meter.style.background = "#ff5050";
            }

            updateDashboard(data, percentage);

        } catch (error) {
            alert("Error connecting to server!");
        }

        btn.textContent = "Analyze Planet";
        btn.disabled = false;
    });

    // -- RESET LOGIC --
    document.getElementById('resetBtn').addEventListener('click', function() {
        renderEarthBaseline();
        const btn = document.getElementById('predictBtn');
        btn.textContent = "Analyze Planet";
        btn.disabled = false;
    });

    function updateDashboard(formData, probabilityPercent) {
        const probability = Math.max(0, Math.min(100, probabilityPercent));
        const prob01 = probability / 100;
        const isHabitable = prob01 >= 0.5;

        // ---- Chart 1: Habitability Gauge (Semi-circle) ----
        Plotly.newPlot('gaugeChart', [{
            type: "indicator",
            mode: "gauge+number",
            value: probability,
            title: { text: "Habitability Probability", font: { color: '#fff', size: 16 } },
            domain: { x: [0, 1], y: [0, 0.5] },
            gauge: {
                shape: 'angular',
                axis: { range: [0, 100], tickcolor: '#d6eaff', tickwidth: 2 },
                bar: { color: '#ffffff' },
                steps: [
                    { range: [0, 40], color: 'rgba(255, 80, 80, 0.4)' },
                    { range: [40, 70], color: 'rgba(255, 215, 0, 0.4)' },
                    { range: [70, 100], color: 'rgba(75, 255, 100, 0.45)' }
                ],
                threshold: {
                    line: { color: '#fff', width: 4 },
                    thickness: 0.8,
                    value: probability
                }
            },
            number: { suffix: "%", font: { color: '#fff', size: 18 } }
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, r: 20, b: 0, l: 20 }
        }, { responsive: true, displayModeBar: false });

        // ---- Chart 2: Decision Boundary ----
        Plotly.newPlot('thresholdChart', [
            {
                type: 'bar',
                orientation: 'h',
                x: [prob01],
                y: ['Habitability Score'],
                marker: {
                    color: isHabitable ? 'rgba(75, 255, 100, 0.8)' : 'rgba(255, 80, 80, 0.8)',
                    line: { color: '#ffffffaa', width: 1 }
                },
                hovertemplate: 'Probability: %{x:.2f}<extra></extra>'
            },
            {
                type: 'scatter',
                mode: 'markers',
                x: [prob01],
                y: ['Marker'],
                marker: { color: '#fff', size: 12, symbol: 'diamond' },
                hoverinfo: 'skip'
            }
        ], {
            title: { text: 'Decision Boundary (0.5)', font: { color: '#fff' } },
            xaxis: {
                range: [0, 1],
                tick0: 0,
                dtick: 0.25,
                tickfont: { color: '#d6eaff' },
                gridcolor: 'rgba(255,255,255,0.1)',
                linecolor: 'rgba(255,255,255,0.2)'
            },
            yaxis: { showticklabels: false },
            shapes: [
                {
                    type: 'line',
                    x0: 0.5,
                    x1: 0.5,
                    y0: -0.5,
                    y1: 1.5,
                    line: { color: '#ffffff', width: 2, dash: 'dash' }
                }
            ],
            annotations: [
                {
                    x: 0.5,
                    y: 0,
                    xref: 'x',
                    yref: 'paper',
                    text: 'Decision Threshold = 0.5',
                    showarrow: true,
                    arrowhead: 4,
                    ax: 0,
                    ay: -30,
                    font: { color: '#fff' }
                }
            ],
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 50, r: 20, b: 40, l: 20 },
            bargap: 0.4
        }, { responsive: true, displayModeBar: false });

        // ---- Chart 3: Planet Radar ----
        const safePeriod = Math.max(formData.P_PERIOD || 0, 0.001);
        const periodNorm = Math.log10(safePeriod + 1) / Math.log10(365 + 1);
        const tempNorm = (formData.P_TEMP_EQUIL || 0) / 288;

        const radarLabels = ['P_RADIUS', 'P_MASS', 'P_GRAVITY', 'P_PERIOD', 'P_TEMP_EQUIL'];
        const planetValues = [
            Number(formData.P_RADIUS) || 0,
            Number(formData.P_MASS) || 0,
            Number(formData.P_GRAVITY) || 0,
            periodNorm,
            tempNorm
        ];
        const earthValues = [1, 1, 1, 1, 1];

        const radarData = [
            {
                type: 'scatterpolar',
                r: [...earthValues, earthValues[0]],
                theta: [...radarLabels, radarLabels[0]],
                fill: 'toself',
                name: 'Earth',
                line: { color: '#4fc3ff' },
                fillcolor: 'rgba(79, 195, 255, 0.25)'
            },
            {
                type: 'scatterpolar',
                r: [...planetValues, planetValues[0]],
                theta: [...radarLabels, radarLabels[0]],
                fill: 'toself',
                name: 'Input Planet',
                line: { color: '#ffb347' },
                fillcolor: 'rgba(255, 179, 71, 0.25)'
            }
        ];

        const maxVal = Math.max(...planetValues, 1.2);

        Plotly.newPlot('radarChart', radarData, {
            title: { text: 'Planet Shape vs Earth', font: { color: '#fff' } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            polar: {
                bgcolor: 'rgba(255,255,255,0.03)',
                radialaxis: {
                    visible: true,
                    range: [0, maxVal],
                    gridcolor: 'rgba(255,255,255,0.1)',
                    linecolor: 'rgba(255,255,255,0.2)',
                    tickfont: { color: '#d6eaff' }
                },
                angularaxis: {
                    gridcolor: 'rgba(255,255,255,0.1)',
                    linecolor: 'rgba(255,255,255,0.2)',
                    tickfont: { color: '#d6eaff' }
                }
            },
            legend: { font: { color: '#fff' } },
            margin: { t: 40, r: 20, b: 20, l: 20 }
        }, { responsive: true, displayModeBar: false });

        // ---- Chart 4: Feature Deviation Bar ----
        const features = [
            { name: 'P_RADIUS', value: Number(formData.P_RADIUS) || 0, earth: 1 },
            { name: 'P_MASS', value: Number(formData.P_MASS) || 0, earth: 1 },
            { name: 'P_GRAVITY', value: Number(formData.P_GRAVITY) || 0, earth: 1 },
            { name: 'P_PERIOD (log norm)', value: periodNorm, earth: 1 },
            { name: 'P_TEMP_EQUIL / 288K', value: tempNorm, earth: 1 }
        ];

        const deviations = features.map(f => ({
            name: f.name,
            dev: (f.value - f.earth) / f.earth
        }));

        const colors = deviations.map(d => d.dev >= 0 ? 'rgba(255, 179, 71, 0.85)' : 'rgba(79, 195, 255, 0.85)');

        Plotly.newPlot('featureChart', [{
            type: 'bar',
            orientation: 'h',
            x: deviations.map(d => d.dev),
            y: deviations.map(d => d.name),
            marker: {
                color: colors,
                line: { color: '#ffffffaa', width: 1 }
            },
            hovertemplate: '%{y}: %{x:.2f}<extra></extra>'
        }], {
            title: { text: 'Feature Deviation from Earth', font: { color: '#fff' } },
            xaxis: {
                zeroline: true,
                zerolinecolor: '#ffffff88',
                tickfont: { color: '#d6eaff' },
                gridcolor: 'rgba(255,255,255,0.1)',
                linecolor: 'rgba(255,255,255,0.2)'
            },
            yaxis: {
                tickfont: { color: '#d6eaff' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 40, r: 20, b: 40, l: 120 }
        }, { responsive: true, displayModeBar: false });

        dashboardGrid.classList.remove('hidden');
    }

    // Initialize dashboard with Earth baseline so charts are never empty on load
    renderEarthBaseline();
});