/**
 * Oprel Message Renderer — Production-Grade ChatGPT/Claude-style rendering
 * Compatible with marked.js v15.x
 *
 * KEY INSIGHT (marked v15):
 *   - Block renderers (paragraph, heading) receive token.text = RAW markdown
 *   - Must call this.parser.parseInline(token.tokens) to render inline formatting
 *   - Inline renderers (strong, em, link) also need this.parser.parseInline(token.tokens)
 *   - code/codespan use token.text directly (no nested tokens)
 *   - blockquote/listitem use this.parser.parse(token.tokens) for block content
 *   - MUST use regular functions (not arrows) so `this` binds to renderer instance
 */

const MessageRenderer = (() => {
    let _initialized = false;

    function _escapeHtml(str) {
        if (typeof str !== 'string') str = String(str || '');
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function init() {
        if (_initialized) return;
        _initialized = true;

        marked.use({
            gfm: true,
            breaks: false,
            async: false,
            renderer: {
                // ━━━ CODE BLOCK (Canvas Component) ━━━━━━━━━━━━━━━
                // token.text = raw code string, token.lang = language
                code(token) {
                    const codeText = token.text || '';
                    const lang = (token.lang || '').trim().split(/\s/)[0] || 'code';
                    const escaped = _escapeHtml(codeText);

                    return '<div class="oprel-code-canvas" data-lang="' + lang + '">'
                        + '<div class="oprel-code-header">'
                        +   '<div class="oprel-code-lang">'
                        +     '<span class="oprel-code-dot"></span>'
                        +     '<span class="oprel-code-dot"></span>'
                        +     '<span class="oprel-code-dot"></span>'
                        +     '<span class="oprel-code-lang-label">' + lang + '</span>'
                        +   '</div>'
                        +   '<div class="oprel-code-actions">'
                        +     '<button class="oprel-code-btn oprel-canvas-btn" onclick="MessageRenderer.openCanvas(this)" title="Open in Canvas">'
                        +       '<iconify-icon icon="solar:maximize-square-bold" width="14"></iconify-icon>'
                        +     '</button>'
                        +     '<button class="oprel-code-btn oprel-copy-btn" onclick="MessageRenderer.copyCode(this)" title="Copy code">'
                        +       '<iconify-icon icon="solar:copy-bold" width="14"></iconify-icon>'
                        +       '<span>Copy</span>'
                        +     '</button>'
                        +   '</div>'
                        + '</div>'
                        + '<div class="oprel-code-body">'
                        +   '<pre><code class="language-' + lang + '">' + escaped + '</code></pre>'
                        + '</div>'
                        + '</div>';
                },

                // ━━━ INLINE CODE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                codespan(token) {
                    return '<code class="oprel-inline-code">' + _escapeHtml(token.text || '') + '</code>';
                },

                // ━━━ HEADING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // MUST use this.parser.parseInline for inline content
                heading(token) {
                    const d = token.depth || 2;
                    const body = this.parser.parseInline(token.tokens);
                    return '<h' + d + ' class="oprel-heading oprel-h' + d + '">' + body + '</h' + d + '>';
                },

                // ━━━ PARAGRAPH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                paragraph(token) {
                    const body = this.parser.parseInline(token.tokens);
                    return '<p class="oprel-paragraph">' + body + '</p>';
                },

                // ━━━ BLOCKQUOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // Uses parse() for block-level children
                blockquote(token) {
                    const body = this.parser.parse(token.tokens);
                    return '<div class="oprel-callout">' + body + '</div>';
                },

                // ━━━ TABLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                table(token) {
                    var html = '<div class="oprel-table-wrap"><table class="oprel-table"><thead><tr>';
                    var self = this;
                    if (token.header) {
                        token.header.forEach(function(cell) {
                            var align = cell.align ? ' style="text-align:' + cell.align + '"' : '';
                            var text = self.parser.parseInline(cell.tokens);
                            html += '<th' + align + '>' + text + '</th>';
                        });
                    }
                    html += '</tr></thead><tbody>';
                    if (token.rows) {
                        token.rows.forEach(function(row) {
                            html += '<tr>';
                            row.forEach(function(cell, i) {
                                var align = (token.header && token.header[i] && token.header[i].align)
                                    ? ' style="text-align:' + token.header[i].align + '"'
                                    : '';
                                var text = self.parser.parseInline(cell.tokens);
                                html += '<td' + align + '>' + text + '</td>';
                            });
                            html += '</tr>';
                        });
                    }
                    html += '</tbody></table></div>';
                    return html;
                },

                // ━━━ LIST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                list(token) {
                    var tag = token.ordered ? 'ol' : 'ul';
                    var startAttr = (token.ordered && token.start !== 1) ? ' start="' + token.start + '"' : '';
                    var html = '<' + tag + ' class="oprel-list"' + startAttr + '>';
                    var self = this;
                    if (token.items) {
                        token.items.forEach(function(item) {
                            html += self.listitem(item);
                        });
                    }
                    html += '</' + tag + '>';
                    return html;
                },

                // ━━━ LIST ITEM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // Uses parse() for block-level content within items
                listitem(token) {
                    var body = '';
                    if (token.tokens) {
                        body = this.parser.parse(token.tokens, !!token.loose);
                    } else {
                        body = token.text || '';
                    }
                    if (token.task) {
                        var checked = token.checked ? ' checked="" disabled=""' : ' disabled=""';
                        body = '<input type="checkbox"' + checked + '> ' + body;
                    }
                    return '<li class="oprel-list-item">' + body + '</li>';
                },

                // ━━━ HR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                hr() {
                    return '<hr class="oprel-divider">';
                },

                // ━━━ LINK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                link(token) {
                    var href = _escapeHtml(token.href || '#');
                    var title = token.title ? ' title="' + _escapeHtml(token.title) + '"' : '';
                    var text = this.parser.parseInline(token.tokens);
                    return '<a href="' + href + '" class="oprel-link" target="_blank" rel="noopener"' + title + '>' + text + '</a>';
                },

                // ━━━ STRONG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                strong(token) {
                    var body = this.parser.parseInline(token.tokens);
                    return '<strong class="oprel-bold">' + body + '</strong>';
                },

                // ━━━ EM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                em(token) {
                    var body = this.parser.parseInline(token.tokens);
                    return '<em class="oprel-italic">' + body + '</em>';
                },
            }
        });
    }

    // ─── Normalize LLM output before parsing ────────────────────────
    // LLMs often omit blank lines between sections. This fixes that
    // so marked.js correctly identifies block-level elements.
    function _normalize(text) {
        if (!text) return '';

        // Normalize line endings
        var out = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

        // Step 1: Ensure blank line BEFORE lines starting with ## heading markers
        out = out.replace(/([^\n])\n(#{1,6} )/g, '$1\n\n$2');

        // Step 2: Ensure blank line BEFORE lines that are bold-only "headings":
        //   **Some Title** followed immediately by text on the same line → split
        //   Pattern: **text** then immediately more text (no newline separation)
        out = out.replace(/(\*\*[^*\n]+\*\*)([ \t]*)([A-Za-z])/g, '$1\n\n$3');

        // Step 3: Ensure blank line BEFORE lines that start with bold (heading-like)
        //   e.g. "paragraph.\n**New Section**" → "paragraph.\n\n**New Section**"
        out = out.replace(/([^\n])\n(\*\*[A-Z])/g, '$1\n\n$2');

        // Step 4: Ensure blank line AFTER lines ending with **heading**
        //   e.g. "**Section**\nBody starts" → "**Section**\n\nBody starts"
        out = out.replace(/(\*\*[^*\n]+\*\*)(\n)([^\n*#\->\s])/g, '$1\n\n$3');

        // Step 5: Ensure blank line before fenced code blocks
        out = out.replace(/([^\n])\n(```)/g, '$1\n\n$2');

        // Step 6: Ensure blank line after fenced code blocks
        out = out.replace(/(```)\n([^\n`])/g, '$1\n\n$2');

        // Step 7: Ensure blank line before unordered list items if preceded by non-list
        out = out.replace(/([^\n\-*+])\n([-*+] )/g, '$1\n\n$2');

        // Step 8: Ensure blank line before numbered list items if preceded by non-list
        out = out.replace(/([^\n\d])\n(\d+\. )/g, '$1\n\n$2');

        // Step 9: Collapse 3+ consecutive blank lines to 2
        out = out.replace(/\n{3,}/g, '\n\n');

        return out.trim();
    }

    // ─── Public: Render markdown to HTML ────────────────────────────
    function render(text) {
        if (!text) return '';
        init();
        try {
            var normalized = _normalize(text);
            var result = marked.parse(normalized);
            return (typeof result === 'string') ? result : String(result || '');
        } catch (e) {
            console.error('MessageRenderer.render error:', e);
            return '<p class="oprel-paragraph">' + _escapeHtml(text) + '</p>';
        }
    }

    // ─── Public: Copy code from a canvas ───────────────────────────
    function copyCode(btn) {
        var canvas = btn.closest('.oprel-code-canvas');
        if (!canvas) return;
        var code = canvas.querySelector('code');
        if (!code) return;

        navigator.clipboard.writeText(code.innerText).then(function() {
            var label = btn.querySelector('span');
            if (label) {
                var original = label.innerText;
                label.innerText = 'Copied!';
                btn.classList.add('oprel-copied');
                setTimeout(function() {
                    label.innerText = original;
                    btn.classList.remove('oprel-copied');
                }, 2000);
            }
        });
    }

    // ─── Public: Open code in Canvas side panel ────────────────────
    function openCanvas(btn) {
        var canvasPanel = document.getElementById('chat-canvas');
        var canvasContent = document.getElementById('canvas-content');
        var canvasTitle = document.getElementById('canvas-title');
        if (!canvasPanel || !canvasContent) return;

        var codeCanvas = btn.closest('.oprel-code-canvas');
        if (!codeCanvas) return;

        var lang = codeCanvas.dataset.lang || 'code';
        var codeEl = codeCanvas.querySelector('code');
        if (!codeEl) return;

        canvasTitle.innerText = lang.toUpperCase() + ' — Artifact';
        canvasContent.innerHTML = '<pre class="oprel-canvas-pre"><code class="language-' + lang + '">' + _escapeHtml(codeEl.innerText) + '</code></pre>';

        var newCodeEl = canvasContent.querySelector('code');
        if (newCodeEl && typeof hljs !== 'undefined') {
            hljs.highlightElement(newCodeEl);
        }

        canvasPanel.classList.remove('hidden');
        canvasPanel.classList.add('flex');
    }

    // ─── Public: Highlight all code blocks in container ────────────
    function highlightAll(container) {
        if (!container || typeof hljs === 'undefined') return;
        container.querySelectorAll('.oprel-code-canvas code[class*="language-"]').forEach(function(block) {
            hljs.highlightElement(block);
        });
    }

    return { init: init, render: render, copyCode: copyCode, openCanvas: openCanvas, highlightAll: highlightAll };
})();
