// script.js
document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  form.addEventListener("submit", function (event) {
    alert("Form submitted!");
    // Prevent actual submit for demonstration
    event.preventDefault();
  });
});
