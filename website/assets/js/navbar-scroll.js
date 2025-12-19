// Smart scroll-aware navbar
(function() {
  'use strict';
  
  let lastScrollTop = 0;
  let ticking = false;
  const navbar = document.querySelector('.navbar-custom');
  const scrollThreshold = 10; // Minimum scroll amount to trigger hide/show
  const navbarHeight = navbar ? navbar.offsetHeight : 0;
  
  function handleScroll() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollingDown = scrollTop > lastScrollTop;
    const atTop = scrollTop <= navbarHeight;
    
    // Check if mobile menu is open
    const mobileMenu = document.getElementById('main-navbar');
    const mobileMenuOpen = mobileMenu && mobileMenu.classList.contains('show');
    
    // Don't hide navbar if at top or mobile menu is open
    if (atTop || mobileMenuOpen) {
      navbar.classList.remove('navbar-hidden');
      navbar.classList.add('navbar-visible');
    } 
    // Hide navbar when scrolling down
    else if (scrollingDown && Math.abs(scrollTop - lastScrollTop) > scrollThreshold) {
      navbar.classList.remove('navbar-visible');
      navbar.classList.add('navbar-hidden');
    } 
    // Show navbar when scrolling up
    else if (!scrollingDown && Math.abs(scrollTop - lastScrollTop) > scrollThreshold) {
      navbar.classList.remove('navbar-hidden');
      navbar.classList.add('navbar-visible');
    }
    
    lastScrollTop = scrollTop;
    ticking = false;
  }
  
  function requestTick() {
    if (!ticking) {
      window.requestAnimationFrame(handleScroll);
      ticking = true;
    }
  }
  
  // Initialize
  if (navbar) {
    navbar.classList.add('navbar-visible');
    window.addEventListener('scroll', requestTick, { passive: true });
    
    // Also watch for mobile menu toggle
    const navbarToggler = document.querySelector('.navbar-toggler');
    if (navbarToggler) {
      navbarToggler.addEventListener('click', function() {
        // Small delay to let Bootstrap finish the toggle
        setTimeout(function() {
          const mobileMenu = document.getElementById('main-navbar');
          const mobileMenuOpen = mobileMenu && mobileMenu.classList.contains('show');
          if (mobileMenuOpen) {
            navbar.classList.remove('navbar-hidden');
            navbar.classList.add('navbar-visible');
          }
        }, 10);
      });
    }
  }
})();
