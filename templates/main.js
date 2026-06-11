document.addEventListener('DOMContentLoaded', () => {
    // Fetch group predictions and tournament winner automatically
    fetch('/api/predict-groups', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        const container = document.getElementById('groups-container');
        if (!container) return;
        
        container.innerHTML = ''; // Clear loading text
        
        // Build group tables HTML
        let delayCounter = 0;
        for (const [groupName, teams] of Object.entries(data.groups)) {
            const tableHtml = `
                <div class="group-table animate-fade-in" style="animation-delay: ${delayCounter * 0.1}s;">
                    <h3>${groupName}</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Team</th>
                                <th>Pts</th>
                                <th>GD</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${teams.map(team => `
                                <tr>
                                    <td class="team-name">${team.team}</td>
                                    <td>${team.points}</td>
                                    <td>${team.gd > 0 ? '+' : ''}${team.gd}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', tableHtml);
            delayCounter++;
        }
        
        // Show winner
        const winnerSection = document.getElementById('winner-section');
        const winnerDisplay = document.getElementById('tournament-winner');
        if (winnerSection && winnerDisplay && data.winner) {
            winnerDisplay.innerHTML = data.winner;
            winnerSection.classList.remove('hidden');
        }
    })
    .catch(error => {
        console.error('Error fetching group predictions:', error);
        const container = document.getElementById('groups-container');
        if (container) {
            container.innerHTML = '<p>Failed to load simulation.</p>';
        }
    });
});
