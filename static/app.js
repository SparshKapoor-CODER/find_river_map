document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("country-form");
  const button = document.getElementById("generate-btn");
  const select = document.getElementById("country-select");

  if (!form || !button || !select) return;

  form.addEventListener("submit", function () {
    const value = select.value;

    if (!value) {
      // rely on HTML5 required validation
      return;
    }

    if (!button.dataset.originalText) {
      button.dataset.originalText = button.textContent;
    }

    button.disabled = true;
    button.textContent = "Generating…";
  });
});
