const movieGrid = document.getElementById("movie-grid");
const sectionTitle = document.getElementById("section-title");
const hero = document.getElementById("hero");

const recommendBtn = document.getElementById("recommend-btn");
const homeBtn = document.getElementById("home-btn");
const trendingBtn = document.getElementById("trending-btn");
const genreBtn = document.getElementById("genre-btn");
const historyBtn = document.getElementById("history-btn");

const navSearchInput = document.getElementById("nav-search-input");
const mainContent = document.getElementById("main-content");
const adminContainer = document.getElementById("admin-container");
const loginPage = document.getElementById("login-page");
const loginBtn = document.getElementById("login-btn");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const loginError = document.getElementById("login-error");

const navbar = document.getElementById("navbar");
const navLinks = document.getElementById("nav-links");
const logoutBtn = document.getElementById("logout-btn");

// Use an empty string for relative paths so it works on both Localhost and Hugging Face
const BACKEND_URL = ""; 
let isLoading = false;

/* ---------------- UI Helpers ---------------- */

function resetView() {
    if(hero) hero.style.display = "block";
    sectionTitle.textContent = "";
    movieGrid.innerHTML = "";
    movieGrid.classList.remove("movie-details-mode");
}

function showSkeletons(count = 8) {
    movieGrid.innerHTML = "";
    for (let i = 0; i < count; i++) {
        const div = document.createElement("div");
        div.className = "movie-card skeleton";
        movieGrid.appendChild(div);
    }
}

function showError(msg) {
    movieGrid.innerHTML = `
        <div style="text-align:center; padding: 2rem; width: 100%;">
            <p style="color: var(--muted);">${msg}</p>
        </div>
    `;
}

function renderMovies(movies, extra = null) {
    movieGrid.innerHTML = movies.map(m => {
        const mid = m.movie_id || m.id;
        const posterUrl = m.poster || m.poster_path || 'https://via.placeholder.com/300x450?text=No+Image';
        const title = m.title || "Unknown Title";
        const year = (m.release_date || m.year || "").toString().slice(0, 4) || "—";
        const tmdbRating = m.rating_tmdb || m.vote_average || "N/A";

        return `
            <div class="movie-card" data-id="${mid}">
                <img src="${posterUrl}">
                <h3>${title}</h3>
                <p class="meta">${year} · ⭐ ${tmdbRating}</p>
                ${extra ? extra(m) : ""}
            </div>
        `;
    }).join("");
}

/* ---------------- API ---------------- */

async function apiGet(path, params = {}) {
    if (isLoading) return null;
    isLoading = true;
    showSkeletons();

    try {
        const res = await axios.get(`${BACKEND_URL}${path}`, { params });
        return res.data;
    } catch (err) {
        console.error("API Error:", err);
        showError("Something went wrong loading data.");
        return null;
    } finally {
        isLoading = false;
    }
}

/* ---------------- Pages ---------------- */

function showHome() {
    resetView();
    const role = sessionStorage.getItem("role");
    if (role === "user") {
        recommendForUser();
    }
}

async function loadTrending() {
    if(hero) hero.style.display = "none";
    sectionTitle.textContent = "Trending Now";
    movieGrid.classList.remove("movie-details-mode");

    const data = await apiGet("/trending", { limit: 20 });
    if (data) renderMovies(data);
}

async function loadGenres() {
    if(hero) hero.style.display = "none";
    sectionTitle.textContent = "Pick a Genre";
    movieGrid.classList.remove("movie-details-mode");

    const genres = ["Action","Comedy","Drama","Romance","Thriller","Animation"];
    movieGrid.innerHTML = genres.map(g =>
        `<button class="nav-btn genre-btn" data-g="${g.toLowerCase()}">${g}</button>`
    ).join("");

    document.querySelectorAll(".genre-btn").forEach(btn => {
        btn.onclick = async () => {
            sectionTitle.textContent = `Genre: ${btn.textContent}`;
            const data = await apiGet("/recommend/genre", { genre: btn.dataset.g, n: 20 });
            if (data) renderMovies(data);
        };
    });
}

async function recommendForUser() {
    const uid = sessionStorage.getItem("user_id");
    if (!uid) return;

    if(hero) hero.style.display = "none";
    sectionTitle.textContent = "Handpicked For You";
    movieGrid.classList.remove("movie-details-mode");

    const data = await apiGet("/recommend", { user_id: Number(uid), n: 12 });
    if (data) renderMovies(data, m =>
        `<p class="score">Taste Match: ${m.predicted_rating ? m.predicted_rating.toFixed(2) : 'N/A'}</p>`
    );
}

async function loadHistory() {
    const uid = sessionStorage.getItem("user_id");
    if (!uid) return;

    if(hero) hero.style.display = "none";
    sectionTitle.textContent = "Your Watch History";
    movieGrid.classList.remove("movie-details-mode");

    const data = await apiGet("/user/history", { user_id: Number(uid) });
    if (!data?.length) return showError("No history found.");
    renderMovies(data, m => `<p class="score">Your Rating: ${m.rating}</p>`);
}

async function searchMovies(q) {
    if (!q) return;
    if(hero) hero.style.display = "none";
    sectionTitle.textContent = `Search: "${q}"`;
    movieGrid.classList.remove("movie-details-mode");

    const data = await apiGet("/search", { query: q });
    if (!data?.length) return showError("No results.");
    renderMovies(data);
}

/* ---------------- Movie Details ---------------- */

movieGrid.addEventListener("click", e => {
    const card = e.target.closest(".movie-card");
    if (!card) return;
    const movieId = card.dataset.id;
    if (!movieId || movieId === "null") return;
    loadMovieDetails(Number(movieId));
});

async function loadMovieDetails(movieId) {
    if(hero) hero.style.display = "none";
    sectionTitle.textContent = "";
    movieGrid.classList.add("movie-details-mode");

    const movie = await apiGet(`/movie/${movieId}`);
    if (!movie) return;

    const similar = await apiGet("/similar", { movie_id: movieId, n: 15 });

    movieGrid.innerHTML = `
        <div class="movie-details-container">
            <button id="back-btn" class="primary-btn">← Back</button>
            <div class="movie-details-card">
                <img src="${movie.poster || 'https://via.placeholder.com/220x330?text=No+Image'}" class="movie-details-poster">
                <div class="movie-details-info">
                    <h2>${movie.title}</h2>
                    <p><strong>Release:</strong> ${movie.release_date || "Unknown"}</p>
                    <p><strong>Rating:</strong> ⭐ ${movie.rating_tmdb ?? "N/A"}</p>
                    <p><strong>Genres:</strong> ${movie.genres || "N/A"}</p>
                    <p><strong>Cast:</strong> ${movie.cast || "N/A"}</p>
                    <p><strong>Overview:</strong> ${movie.overview || "N/A"}</p>
                </div>
            </div>
            ${similar?.length ? `
            <div class="similar-section">
                <h3 class="similar-title">Similar Movies</h3>
                <div class="similar-carousel-wrapper">
                    <button class="carousel-btn left">&lt;</button>
                    <div class="similar-movies-carousel">
                        ${similar.map(m => `
                            <div class="similar-movie-card" data-id="${m.movie_id || m.id}">
                                <img src="${m.poster || 'https://via.placeholder.com/200x300?text=No+Image'}">
                                <p>${m.title}</p>
                            </div>
                        `).join("")}
                    </div>
                    <button class="carousel-btn right">&gt;</button>
                </div>
            </div>
            ` : ""}
        </div>
    `;

    document.getElementById("back-btn").onclick = showHome;

    document.querySelectorAll(".similar-movie-card").forEach(card => {
        card.onclick = () => loadMovieDetails(Number(card.dataset.id));
    });

    const carousel = document.querySelector(".similar-movies-carousel");
    if (carousel) {
        document.querySelector(".carousel-btn.left").onclick = () => carousel.scrollBy({ left: -500, behavior: "smooth" });
        document.querySelector(".carousel-btn.right").onclick = () => carousel.scrollBy({ left: 500, behavior: "smooth" });
    }
}

/* ---------------- Auth & Session ---------------- */

function logout() {
    sessionStorage.clear();
    window.location.reload();
}
if(logoutBtn) logoutBtn.onclick = logout;

function configureNavbarForRole(role) {
    if(!navbar) return;
    navbar.style.display = "flex";
    if(logoutBtn) logoutBtn.style.display = "block";

    const displayStyle = (role === "admin") ? "none" : "flex";
    
    if(homeBtn) homeBtn.style.display = displayStyle;
    if(trendingBtn) trendingBtn.style.display = displayStyle;
    if(genreBtn) genreBtn.style.display = displayStyle;
    if(historyBtn) historyBtn.style.display = displayStyle;
    if(navSearchInput) navSearchInput.parentElement.style.display = displayStyle;
}

loginBtn?.addEventListener("click", async () => {
    const usernameInputVal = usernameInput.value.trim();
    const password = passwordInput.value;

    if (!usernameInputVal || !password) {
        loginError.textContent = "Please enter username and password.";
        return;
    }

    try {
        const res = await axios.post(`${BACKEND_URL}/login`, { username: usernameInputVal, password });
        const { role, user_id, username } = res.data;
        
        sessionStorage.setItem("user_id", user_id);
        sessionStorage.setItem("role", role);
        sessionStorage.setItem("username", username); 

        loginPage.style.display = "none"; 
        configureNavbarForRole(role);

        if (role === "admin") {
            mainContent.style.display = "none";
            adminContainer.style.display = "block";
            loadAdminMetrics(); 
        } else {
            adminContainer.style.display = "none";
            mainContent.style.display = "block";
            showHome();
        }
    } catch (err) {
        loginError.textContent = "Invalid username or password.";
    }
});

window.addEventListener("DOMContentLoaded", () => {
    const role = sessionStorage.getItem("role");
    const userId = sessionStorage.getItem("user_id");

    // Initial state
    if(loginPage) loginPage.style.display = "none";
    if(mainContent) mainContent.style.display = "none";
    if(adminContainer) adminContainer.style.display = "none";
    if(navbar) navbar.style.display = "none";

    if (role === "admin") {
        configureNavbarForRole("admin");
        adminContainer.style.display = "block";
        loadAdminMetrics();
    } else if (role === "user" && userId) {
        configureNavbarForRole("user");
        mainContent.style.display = "block";
        showHome();
    } else {
        if(loginPage) loginPage.style.display = "block";
        sessionStorage.clear(); 
    }
});

/* ---------------- Admin Logic ---------------- */

async function loadAdminMetrics() {
    const role = sessionStorage.getItem("role");
    const username = sessionStorage.getItem("username");
    if (role !== "admin" || !username) return;

    try {
        const res = await axios.get(`${BACKEND_URL}/admin/stats`, { params: { username } });
        const data = res.data;
        document.querySelector("#total-users p").textContent = data.total_users;
        document.querySelector("#total-movies p").textContent = data.total_movies;
        document.querySelector("#total-ratings p").textContent = data.total_ratings;
        document.querySelector("#recent-activity p").textContent = data.recent_ratings || "—";

        const tbody = document.getElementById("admin-table-body");
        tbody.innerHTML = data.user_metrics.map(u => `
            <tr>
                <td>${u.user_id}</td>
                <td>${u.ratings_count}</td>
                <td>${u.avg_rating.toFixed(2)}</td>
                <td>${u.last_activity ?? "—"}</td>
            </tr>
        `).join("");
    } catch (err) {
        console.error("Admin metrics load failed:", err);
    }
}

/* ---------------- Events ---------------- */

if(homeBtn) homeBtn.onclick = showHome;
if(trendingBtn) trendingBtn.onclick = loadTrending;
if(genreBtn) genreBtn.onclick = loadGenres;
if(historyBtn) historyBtn.onclick = loadHistory;

if(navSearchInput) {
    navSearchInput.addEventListener("keydown", e => {
        if (e.key === "Enter") searchMovies(navSearchInput.value);
    });
}