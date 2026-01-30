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
            }
        });
    });

    // Render equation with KaTeX and highlighting
    if (typeof katex !== 'undefined') {
        const eq1 = document.getElementById('eq1');
        if (eq1) {
            // Build formula with separate spans grouped into logical blocks
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

            // Render main KaTeX parts
            katex.render('\\hat{d}(\\mathbf{x})', eq1.querySelector('[data-hl="dhat"]'), { throwOnError: false });
            katex.render('\\displaystyle\\sum_{k=1}^{K}', eq1.querySelector('[data-hl="sum"]'), { throwOnError: false });
            katex.render('w_k', eq1.querySelector('[data-hl="wk"]'), { throwOnError: false });
            katex.render('\\tfrac{1}{2}', eq1.querySelector('.frac'), { throwOnError: false });

            // Render mu and sigma (there are two mu spans)
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

    // Run after KaTeX renders
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

    // Segmented toggle
    const viewToggle = document.getElementById('viewToggle');
    const panelEdf = document.getElementById('panel-edf');
    const panelGradient = document.getElementById('panel-gradient');
    const options = document.querySelectorAll('.segment-option');
    const indicator = document.getElementById('segmentIndicator');

    function updateIndicator(activeOption) {
        if (indicator && activeOption) {
            indicator.style.width = activeOption.offsetWidth + 'px';
            indicator.style.left = activeOption.offsetLeft + 'px';
        }
    }

    function setActiveOption(isGradient) {
        options.forEach(opt => opt.classList.remove('active-text'));
        if (isGradient) {
            options[1].classList.add('active-text');
            updateIndicator(options[1]);
            panelEdf.classList.remove('active');
            panelGradient.classList.add('active');
        } else {
            options[0].classList.add('active-text');
            updateIndicator(options[0]);
            panelEdf.classList.add('active');
            panelGradient.classList.remove('active');
        }
    }

    if (viewToggle && options.length >= 2) {
        // Initial position
        setTimeout(() => updateIndicator(options[0]), 10);

        // Click on options
        options.forEach(opt => {
            opt.addEventListener('click', function () {
                const isGradient = this.dataset.value === 'gradient';
                viewToggle.checked = isGradient;
                setActiveOption(isGradient);
            });
        });

        // Checkbox change
        viewToggle.addEventListener('change', function () {
            setActiveOption(this.checked);
        });
    }
});
