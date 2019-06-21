window.MathJax = {
    config: ["MMLorHTML.js"],
    jax: ["input/TeX", "output/HTML-CSS", "output/NativeMML"],
    extensions: ["MathMenu.js", "MathZoom.js"],
    tex2jax: {
        inlineMath: [
            ["$", "$"],
            ["\\(", "\\)"]
        ],
        displayMath: [
            ["$$", "$$"],
            ["\\[", "\\]"]
        ]
    },
    TeX: {
        TagSide: "right",
        TagIndent: ".8em",
        MultLineWidth: "85%",
        equationNumbers: {
            autoNumber: "AMS",
        },
        unicode: {
            fonts: "STIXGeneral,'Arial Unicode MS'"
        }
    },
    displayAlign: "center",
    showProcessingMessages: false,
    messageStyle: "none"
};
