// G-EDF Project Website - JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    const savedTheme = localStorage.getItem('theme');
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');

    document.documentElement.setAttribute('data-theme', initialTheme);

    themeToggle.addEventListener('click', function () {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;

            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });

                // Close mobile menu if open
                const navLinks = document.getElementById('navLinks');
                const hamburger = document.getElementById('hamburgerBtn');
                if (navLinks && navLinks.classList.contains('open')) {
                    navLinks.classList.remove('open');
                    hamburger.classList.remove('active');
                }
            }
        });
    });

    // Render equation with KaTeX and highlighting
    if (typeof katex !== 'undefined') {
        const eq1 = document.getElementById('eq1');
        if (eq1) {
            eq1.innerHTML = `
                <span class="eq-block">
                    <span class="hl-formula" data-hl="dhat"></span>
                    <span class="eq-sep">=</span>
                    <span class="hl-formula" data-hl="sum"></span>
                    <span class="hl-formula" data-hl="wk"></span>
                </span>
                <span class="eq-block">
                    <span class="eq-part">exp(−</span>
                    <span class="eq-part frac"></span>
                    <span class="eq-part">(x −</span>
                    <span class="hl-formula" data-hl="mu"></span>
                    <span class="eq-part">)ᵀ</span>
                    <span class="hl-formula" data-hl="sigma"></span>
                    <span class="eq-part">⁻¹(x −</span>
                    <span class="hl-formula" data-hl="mu"></span>
                    <span class="eq-part">))</span>
                </span>
            `;

            katex.render('\\hat{d}(\\mathbf{x})', eq1.querySelector('[data-hl="dhat"]'), { throwOnError: false });
            katex.render('\\displaystyle\\sum_{k=1}^{K}', eq1.querySelector('[data-hl="sum"]'), { throwOnError: false });
            katex.render('w_k', eq1.querySelector('[data-hl="wk"]'), { throwOnError: false });
            katex.render('\\tfrac{1}{2}', eq1.querySelector('.frac'), { throwOnError: false });

            const muSpans = eq1.querySelectorAll('[data-hl="mu"]');
            muSpans.forEach(span => {
                katex.render('\\boldsymbol{\\mu}_k', span, { throwOnError: false });
            });
            katex.render('\\boldsymbol{\\Sigma}_k', eq1.querySelector('[data-hl="sigma"]'), { throwOnError: false });
        }
    }

    // Formula hover highlighting - bidirectional
    function setupHighlighting() {
        const allHlElements = document.querySelectorAll('[data-highlight], [data-hl]');

        allHlElements.forEach(el => {
            el.addEventListener('mouseenter', function () {
                const hlType = this.dataset.highlight || this.dataset.hl;
                document.querySelectorAll(`[data-highlight="${hlType}"], [data-hl="${hlType}"]`).forEach(target => {
                    target.classList.add('hl-active');
                });
            });

            el.addEventListener('mouseleave', function () {
                const hlType = this.dataset.highlight || this.dataset.hl;
                document.querySelectorAll(`[data-highlight="${hlType}"], [data-hl="${hlType}"]`).forEach(target => {
                    target.classList.remove('hl-active');
                });
            });
        });
    }

    setTimeout(setupHighlighting, 100);

    // Copy button functionality
    const copyBtn = document.getElementById('copyBtn');
    const installCode = document.getElementById('installCode');

    if (copyBtn && installCode) {
        copyBtn.addEventListener('click', async function () {
            try {
                await navigator.clipboard.writeText(installCode.textContent);
                const copyText = this.querySelector('.copy-text');
                copyText.textContent = 'Copied!';
                this.classList.add('copied');

                setTimeout(() => {
                    copyText.textContent = 'Copy';
                    this.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });
    }

    // Segmented toggles (generic: works for any .segmented-toggle container)
    function initSegmentedToggle(containerId, checkboxId, indicatorId, panelIds) {
        const container = document.getElementById(containerId);
        const checkbox = document.getElementById(checkboxId);
        const indicator = document.getElementById(indicatorId);
        if (!container || !checkbox || !indicator) return;

        const options = container.querySelectorAll('.segment-option');
        const panels = panelIds.map(id => document.getElementById(id));
        if (options.length < 2 || panels.some(p => !p)) return;

        function updateIndicator(activeOption) {
            indicator.style.width = activeOption.offsetWidth + 'px';
            indicator.style.left = activeOption.offsetLeft + 'px';
        }

        function setActiveOption(index) {
            options.forEach(opt => opt.classList.remove('active-text'));
            panels.forEach(panel => panel.classList.remove('active'));
            options[index].classList.add('active-text');
            updateIndicator(options[index]);
            panels[index].classList.add('active');
        }

        setTimeout(() => updateIndicator(options[0]), 10);

        options.forEach((opt, index) => {
            opt.addEventListener('click', function () {
                checkbox.checked = index === 1;
                setActiveOption(index);
            });
        });

        checkbox.addEventListener('change', function () {
            setActiveOption(this.checked ? 1 : 0);
        });
    }

    initSegmentedToggle('segmentedToggle', 'viewToggle', 'segmentIndicator', ['panel-edf', 'panel-gradient']);
    initSegmentedToggle('trajToggle', 'trajViewToggle', 'trajSegmentIndicator', ['panel-traj-bc', 'panel-traj-park']);

    // ========================================
    // Robustness Comparison (trajectory + condition selector -> compact table)
    // ========================================
    (function initRobustnessComparison() {
        const dataEl = document.getElementById('robustnessData');
        const trajToggle = document.getElementById('resultTrajToggle');
        const condToggle = document.getElementById('resultCondToggle');
        const trajIndicator = document.getElementById('resultTrajIndicator');
        const condIndicator = document.getElementById('resultCondIndicator');
        const tbody = document.getElementById('robustnessTbody');
        const caption = document.getElementById('robustnessCaption');
        if (!dataEl || !trajToggle || !condToggle || !tbody) return;

        const data = JSON.parse(dataEl.textContent);
        const condLabels = { inertial: 'Inertial', 'no-imu': 'No IMU', low: 'Low Noise', high: 'High Noise' };

        let activeTraj = 'bc';
        let activeCond = 'inertial';

        function fmt(value, flag) {
            if (flag === 'b') return `<strong>${value}</strong>`;
            if (flag === 'u') return `<u>${value}</u>`;
            return `${value}`;
        }

        function render() {
            const rows = data[activeTraj][activeCond];
            tbody.innerHTML = rows.map(([method, pos, rot, time, posFlag, rotFlag, timeFlag]) => `
                <tr>
                    <td>${method}</td>
                    <td>${fmt(pos, posFlag)}</td>
                    <td>${fmt(rot, rotFlag)}</td>
                    <td>${fmt(time, timeFlag)}</td>
                </tr>
            `).join('');
            caption.innerHTML = `${activeTraj} &mdash; ${condLabels[activeCond]}. Best results in <strong>bold</strong>, second-best <u>underlined</u>.`;
        }

        function initSelector(toggle, indicator, onSelect) {
            const options = toggle.querySelectorAll('.segment-option');

            function updateIndicator(opt) {
                indicator.style.width = opt.offsetWidth + 'px';
                indicator.style.left = opt.offsetLeft + 'px';
            }

            function selectOption(opt) {
                options.forEach(o => o.classList.remove('active-text'));
                opt.classList.add('active-text');
                updateIndicator(opt);
                onSelect(opt.dataset.value);
                render();
            }

            options.forEach(opt => {
                opt.addEventListener('click', () => selectOption(opt));
            });

            setTimeout(() => updateIndicator(toggle.querySelector('.segment-option.active-text')), 10);
        }

        initSelector(trajToggle, trajIndicator, value => { activeTraj = value; });
        initSelector(condToggle, condIndicator, value => { activeCond = value; });

        render();
    })();

    // ========================================
    // Image Carousel (Snail dataset)
    // ========================================
    const slides = document.querySelectorAll('.carousel-slide');
    const dots = document.querySelectorAll('.carousel-dot');
    const prevBtn = document.getElementById('carouselPrev');
    const nextBtn = document.getElementById('carouselNext');
    let currentSlide = 0;
    let autoplayTimer = null;

    function goToSlide(index) {
        if (slides.length === 0) return;
        slides[currentSlide].classList.remove('active');
        dots[currentSlide].classList.remove('active');
        currentSlide = (index + slides.length) % slides.length;
        slides[currentSlide].classList.add('active');
        dots[currentSlide].classList.add('active');

        // Update caption
        const captionObj = document.getElementById('carouselCaption');
        if (captionObj) {
            captionObj.textContent = slides[currentSlide].dataset.caption || '';
        }
    }

    function nextSlide() {
        goToSlide(currentSlide + 1);
    }

    function prevSlide() {
        goToSlide(currentSlide - 1);
    }

    function startAutoplay() {
        stopAutoplay();
        autoplayTimer = setInterval(nextSlide, 4500);
    }

    function stopAutoplay() {
        if (autoplayTimer) {
            clearInterval(autoplayTimer);
            autoplayTimer = null;
        }
    }

    if (prevBtn && nextBtn) {
        prevBtn.addEventListener('click', () => {
            prevSlide();
            startAutoplay(); // Reset autoplay on interaction
        });
        nextBtn.addEventListener('click', () => {
            nextSlide();
            startAutoplay();
        });
    }

    dots.forEach(dot => {
        dot.addEventListener('click', function () {
            goToSlide(parseInt(this.dataset.index));
            startAutoplay();
        });
    });

    // Start autoplay
    if (slides.length > 1) {
        startAutoplay();
    }

    // ========================================
    // Hamburger Mobile Menu
    // ========================================
    const hamburger = document.getElementById('hamburgerBtn');
    const navLinks = document.getElementById('navLinks');

    if (hamburger && navLinks) {
        hamburger.addEventListener('click', function () {
            this.classList.toggle('active');
            navLinks.classList.toggle('open');
        });

        // Close on outside click
        document.addEventListener('click', function (e) {
            if (navLinks.classList.contains('open') &&
                !navLinks.contains(e.target) &&
                !hamburger.contains(e.target)) {
                navLinks.classList.remove('open');
                hamburger.classList.remove('active');
            }
        });
    }

    // ========================================
    // Scroll-Reveal Animation
    // ========================================
    const revealElements = document.querySelectorAll('[data-reveal]');

    if (revealElements.length > 0) {
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.12,
            rootMargin: '0px 0px -40px 0px'
        });

        revealElements.forEach(el => revealObserver.observe(el));
    }

    // ========================================
    // Lightbox
    // ========================================
    const lightboxOverlay = document.getElementById('lightboxOverlay');
    const lightboxImg = document.getElementById('lightboxImg');
    const lightboxClose = document.getElementById('lightboxClose');

    if (lightboxOverlay && lightboxImg) {
        // Make all result images and carousel images clickable
        const clickableImages = document.querySelectorAll('.images-row img, .carousel-slide img');

        clickableImages.forEach(img => {
            img.style.cursor = 'zoom-in';
            img.addEventListener('click', function () {
                lightboxImg.src = this.src;
                lightboxImg.alt = this.alt;
                lightboxOverlay.classList.add('active');
                document.body.style.overflow = 'hidden';
            });
        });

        function closeLightbox() {
            lightboxOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }

        lightboxClose.addEventListener('click', closeLightbox);
        lightboxOverlay.addEventListener('click', function (e) {
            if (e.target === this) closeLightbox();
        });

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && lightboxOverlay.classList.contains('active')) {
                closeLightbox();
            }
        });
    }

    // ========================================
    // Citation Copy
    // ========================================

    // Copy buttons for citations
    const citationCopyBtns = document.querySelectorAll('.citation-block .copy-btn');

    citationCopyBtns.forEach(btn => {
        btn.addEventListener('click', async function () {
            const block = this.closest('.citation-block');
            const codeContent = block.querySelector('.cite-content').textContent;

            try {
                await navigator.clipboard.writeText(codeContent);
                const copyText = this.querySelector('.copy-text');
                copyText.textContent = 'Copied!';
                this.classList.add('copied');

                setTimeout(() => {
                    copyText.textContent = 'Copy';
                    this.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });
    });
});

