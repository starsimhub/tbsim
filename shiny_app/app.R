# TBsim Shiny Web Application
# A web interface for running tuberculosis simulations using the tbsim package

library(shiny)
library(plotly)
library(DT)
library(reticulate)
library(shinydashboard)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

# Import required Python modules
tbsim <- import("tbsim")
starsim <- import("starsim")
sciris <- import("sciris")
matplotlib <- import("matplotlib")
matplotlib$use("Agg")  # Use non-interactive backend for web
plt <- import("matplotlib.pyplot")
np <- import("numpy")
pd <- import("pandas")

# Define UI
ui <- fluidPage(
  # Enhanced Dark theme CSS
  tags$head(
    tags$style(HTML("
      .dark-theme {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
        color: #e8e8e8 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
      }
      .dark-theme .sidebar {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d3e 100%) !important;
        color: #e8e8e8 !important;
        border-right: 1px solid #3a3a4a !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.3) !important;
      }
      .dark-theme .main-panel {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
        color: #e8e8e8 !important;
      }
      .dark-theme .panel {
        background: rgba(30, 30, 46, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px) !important;
      }
      .dark-theme .form-control {
        background: rgba(45, 45, 62, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid #4a4a5a !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
      }
      .dark-theme .form-control:focus {
        background: rgba(55, 55, 72, 0.9) !important;
        border-color: #007bff !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
      }
      .dark-theme .btn {
        background: linear-gradient(45deg, #4a4a5a, #5a5a6a) !important;
        color: #ffffff !important;
        border: 1px solid #6a6a7a !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
      }
      .dark-theme .btn:hover {
        background: linear-gradient(45deg, #5a5a6a, #6a6a7a) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
      }
      .dark-theme .btn-primary {
        background: linear-gradient(45deg, #007bff, #0056b3) !important;
        border-color: #007bff !important;
      }
      .dark-theme .btn-primary:hover {
        background: linear-gradient(45deg, #0056b3, #004085) !important;
      }
      .dark-theme .btn-secondary {
        background: linear-gradient(45deg, #6c757d, #5a6268) !important;
        border-color: #6c757d !important;
      }
      .dark-theme .btn-secondary:hover {
        background: linear-gradient(45deg, #5a6268, #495057) !important;
      }
      .dark-theme .table {
        background: rgba(30, 30, 46, 0.8) !important;
        color: #e8e8e8 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
      }
      .dark-theme .table-striped > tbody > tr:nth-of-type(odd) {
        background: rgba(45, 45, 62, 0.6) !important;
      }
      .dark-theme .table-striped > tbody > tr:hover {
        background: rgba(0, 123, 255, 0.1) !important;
        transition: background 0.3s ease !important;
      }
      .dark-theme .dataTables_wrapper {
        background: rgba(30, 30, 46, 0.8) !important;
        color: #e8e8e8 !important;
        border-radius: 8px !important;
        padding: 15px !important;
      }
      .dark-theme .dataTables_filter input {
        background: rgba(45, 45, 62, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid #4a4a5a !important;
        border-radius: 6px !important;
      }
      .dark-theme .dataTables_length select {
        background: rgba(45, 45, 62, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid #4a4a5a !important;
        border-radius: 6px !important;
      }
      .dark-theme .dataTables_info {
        color: #b8b8c8 !important;
      }
      .dark-theme .dataTables_paginate .paginate_button {
        background: rgba(45, 45, 62, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid #4a4a5a !important;
        border-radius: 6px !important;
        margin: 0 2px !important;
        transition: all 0.3s ease !important;
      }
      .dark-theme .dataTables_paginate .paginate_button:hover {
        background: rgba(0, 123, 255, 0.2) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
      }
      .dark-theme .dataTables_paginate .paginate_button.current {
        background: linear-gradient(45deg, #007bff, #0056b3) !important;
        color: #ffffff !important;
        border-color: #007bff !important;
      }
      .dark-theme .tab-content {
        background: rgba(15, 15, 35, 0.5) !important;
        color: #e8e8e8 !important;
        border-radius: 8px !important;
        padding: 20px !important;
      }
      .dark-theme .nav-tabs {
        border-bottom: 2px solid #3a3a4a !important;
        background: rgba(30, 30, 46, 0.8) !important;
        border-radius: 8px 8px 0 0 !important;
      }
      .dark-theme .nav-tabs .nav-link {
        color: #b8b8c8 !important;
        background: rgba(45, 45, 62, 0.6) !important;
        border: 1px solid #4a4a5a !important;
        border-radius: 6px 6px 0 0 !important;
        margin-right: 5px !important;
        transition: all 0.3s ease !important;
      }
      .dark-theme .nav-tabs .nav-link:hover {
        background: rgba(0, 123, 255, 0.1) !important;
        color: #ffffff !important;
        border-color: #007bff !important;
        transform: translateY(-2px) !important;
      }
      .dark-theme .nav-tabs .nav-link.active {
        background: linear-gradient(45deg, #007bff, #0056b3) !important;
        color: #ffffff !important;
        border-color: #007bff !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3) !important;
      }
      .dark-theme #header {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3e 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
        border: 1px solid #3a3a4a !important;
      }
      .dark-theme #header h2 {
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
      }
      .dark-theme .well {
        background: rgba(30, 30, 46, 0.8) !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
      }
      .dark-theme .slider-selection {
        background: linear-gradient(45deg, #007bff, #0056b3) !important;
      }
      .dark-theme .slider-track-high {
        background: rgba(45, 45, 62, 0.8) !important;
      }
      .dark-theme .slider-handle {
        background: linear-gradient(45deg, #007bff, #0056b3) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
      }
      .dark-theme .checkbox {
        color: #e8e8e8 !important;
      }
      .dark-theme .checkbox input[type='checkbox']:checked + span {
        color: #007bff !important;
      }
      .dark-theme h1, .dark-theme h2, .dark-theme h3, .dark-theme h4 {
        color: #ffffff !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
      }
      .dark-theme .text-muted {
        color: #b8b8c8 !important;
      }
      .dark-theme .alert {
        background: rgba(30, 30, 46, 0.9) !important;
        border: 1px solid #3a3a4a !important;
        color: #e8e8e8 !important;
        border-radius: 8px !important;
      }
      .dark-theme .alert-info {
        background: rgba(0, 123, 255, 0.1) !important;
        border-color: #007bff !important;
        color: #b8d4ff !important;
      }
      .dark-theme .alert-success {
        background: rgba(40, 167, 69, 0.1) !important;
        border-color: #28a745 !important;
        color: #b8e6c1 !important;
      }
      .dark-theme .alert-warning {
        background: rgba(255, 193, 7, 0.1) !important;
        border-color: #ffc107 !important;
        color: #fff3cd !important;
      }
      .dark-theme .alert-danger {
        background: rgba(220, 53, 69, 0.1) !important;
        border-color: #dc3545 !important;
        color: #f8d7da !important;
      }
    "))
  ),
  
  # JavaScript for theme switching
  tags$script(HTML("
    $(document).ready(function() {
      // Check for saved theme preference
      var savedTheme = localStorage.getItem('darkTheme');
      if (savedTheme === 'true') {
        $('body').addClass('dark-theme');
        $('#dark_theme').prop('checked', true);
      }
      
      // Theme toggle functionality
      $('#dark_theme').on('change', function() {
        if ($(this).is(':checked')) {
          $('body').addClass('dark-theme');
          localStorage.setItem('darkTheme', 'true');
        } else {
          $('body').removeClass('dark-theme');
          localStorage.setItem('darkTheme', 'false');
        }
      });
    });
  ")),
  
  # Enhanced header with logo and theme toggle
  div(
    id = "header",
    style = "display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px; 
             padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
             border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
             border: 1px solid #dee2e6; transition: all 0.3s ease;",
    div(
      style = "display: flex; align-items: center;",
      img(src = "logo.png", height = 60, style = "margin-right: 20px; vertical-align: middle; 
           border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"),
      h2("TBsim - Tuberculosis Simulation Web Interface", 
         style = "margin: 0; color: #2c3e50; font-weight: 600; 
                  text-shadow: 0 1px 2px rgba(0,0,0,0.1);")
    ),
    div(
      style = "display: flex; align-items: center; gap: 15px; padding: 8px 15px; 
               background: rgba(255,255,255,0.8); border-radius: 25px; 
               border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
      span("🌙", style = "font-size: 20px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.2));"),
      div(
        checkboxInput("dark_theme", "Dark Theme", value = FALSE),
        style = "margin: 0; transform: scale(1.1);"
      ),
      span("☀️", style = "font-size: 20px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.2));")
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
        # Action buttons at the top - side by side
        div(
          div(
            actionButton("run_simulation", "Run Simulation", class = "btn-primary btn-lg"),
            style = "display: inline-block; margin-right: 10px;"
          ),
          div(
            actionButton("reset_params", "Reset to Defaults", class = "btn-secondary"),
            style = "display: inline-block;"
          )
        ),
        br(), br(),
      h3("Simulation Parameters"),
      
      # Basic simulation parameters
      h4("Basic Settings"),
      sliderInput("n_agents", "Population Size", value = 1000, min = 100, max = 10000, step = 100),
      fluidRow(
        column(4, dateInput("start_date", "Start Date", value = "1940-01-01")),
        column(4, dateInput("end_date", "End Date", value = "2010-12-31", max = Sys.Date() + 365*50)),
        column(4, numericInput("rand_seed", "Random Seed", value = 1, min = 1, max = 10000, step = 1))
      ),
      sliderInput("dt", "Time Step (days)", value = 7, min = 1, max = 30, step = 1),
      
      # TB-specific parameters
      h4("TB Disease Parameters"),
      sliderInput("init_prev", "Initial Prevalence", value = 0.05, min = 0, max = 1, step = 0.001),
      sliderInput("beta", "Transmission Rate (per year)", value = 0.1, min = 0, max = 1, step = 0.01),
      sliderInput("p_latent_fast", "Probability of Fast Latent TB", value = 0.1, min = 0, max = 1, step = 0.01),
      
      # TB State Transition Rates
      h4("TB State Transition Rates"),
      sliderInput("rate_LS_to_presym", "Latent Slow → Pre-symptomatic (per day)", value = 0.001, min = 0, max = 0.1, step = 0.0001),
      sliderInput("rate_LF_to_presym", "Latent Fast → Pre-symptomatic (per day)", value = 0.01, min = 0, max = 0.1, step = 0.001),
      sliderInput("rate_presym_to_active", "Pre-symptomatic → Active (per day)", value = 0.1, min = 0, max = 1, step = 0.01),
      sliderInput("rate_active_to_clear", "Active → Clearance (per day)", value = 0.01, min = 0, max = 0.1, step = 0.001),
      sliderInput("rate_treatment_to_clear", "Treatment → Clearance (per year)", value = 6, min = 0, max = 50, step = 1),
      
      # TB Mortality Rates
      h4("TB Mortality Rates"),
      sliderInput("rate_exptb_to_dead", "Extra-Pulmonary TB → Death (per day)", value = 0.15 * 4.5e-4, min = 0, max = 1e-3, step = 1e-6),
      sliderInput("rate_smpos_to_dead", "Smear Positive → Death (per day)", value = 4.5e-4, min = 0, max = 1e-3, step = 1e-6),
      sliderInput("rate_smneg_to_dead", "Smear Negative → Death (per day)", value = 0.3 * 4.5e-4, min = 0, max = 1e-3, step = 1e-6),
      
      # TB Transmissibility
      h4("TB Transmissibility"),
      sliderInput("rel_trans_presymp", "Pre-symptomatic Relative Transmissibility", value = 0.1, min = 0, max = 1, step = 0.01),
      sliderInput("rel_trans_smpos", "Smear Positive Relative Transmissibility", value = 1.0, min = 0, max = 2, step = 0.1),
      sliderInput("rel_trans_smneg", "Smear Negative Relative Transmissibility", value = 0.3, min = 0, max = 1, step = 0.01),
      sliderInput("rel_trans_exptb", "Extra-Pulmonary Relative Transmissibility", value = 0.05, min = 0, max = 1, step = 0.01),
      sliderInput("rel_trans_treatment", "Treatment Effect on Transmissibility", value = 0.5, min = 0, max = 1, step = 0.01),
      
      # TB Susceptibility
      h4("TB Susceptibility"),
      sliderInput("rel_sus_latentslow", "Latent Slow Relative Susceptibility", value = 0.20, min = 0, max = 1, step = 0.01),
      
      
      # Demographics
      h4("Demographics"),
      sliderInput("birth_rate", "Birth Rate (per 1000)", value = 20, min = 0, max = 100, step = 1),
      sliderInput("death_rate", "Death Rate (per 1000)", value = 15, min = 0, max = 100, step = 1),
      
      # Network parameters
      h4("Social Network"),
      sliderInput("n_contacts", "Average Contacts per Person", value = 5, min = 1, max = 50, step = 1),
      
      
      # Status
      br(), br(),
      verbatimTextOutput("status")
    ),
    
    mainPanel(
      tabsetPanel(
        id = "main_tabs",
        
        # Results tab
        tabPanel("Results", 
          h3("Simulation Results"),
          uiOutput("loading_spinner"),
          div(style = "margin-bottom: 10px;",
            selectInput("plot_scale", "Y-Axis Scale", 
                       choices = list("Linear Scale" = "linear", "Logarithmic Scale" = "log"),
                       selected = "log", width = "200px")
          ),
          plotlyOutput("results_plot", height = "600px"),
          br(),
          h4("Summary Statistics"),
          DT::dataTableOutput("summary_table"),
          br(),
          h4("Simulation Parameters Used"),
          DT::dataTableOutput("parameters_table")
        ),
        
        # Plots tab
        tabPanel("Detailed Plots",
          h3("Detailed Analysis"),
          plotlyOutput("detailed_plot", height = "600px"),
          br(),
          h4("State Transitions"),
          plotlyOutput("transitions_plot", height = "400px")
        ),
        
        # Advanced Visualizations tab
        tabPanel("Advanced Visualizations",
          h3("Advanced TB Analysis"),
          fluidRow(
            column(6,
              h4("📊 Disease Prevalence Over Time"),
              plotlyOutput("prevalence_plot", height = "400px")
            ),
            column(6,
              h4("🔄 Transmission Flow"),
              plotlyOutput("transmission_sankey", height = "600px")
            )
          ),
          br(),
          fluidRow(
            column(12,
              h4("📈 Interactive Time Series"),
              plotlyOutput("interactive_timeseries", height = "500px")
            )
          ),
          br(),
          fluidRow(
            column(6,
              h4("🎯 Disease Progression"),
              plotlyOutput("disease_progression", height = "400px")
            ),
            column(6,
              h4("📉 Mortality Analysis"),
              plotlyOutput("mortality_analysis", height = "400px")
            )
          )
        ),
        
        # Data tab
        tabPanel("Raw Data",
          h3("Raw Simulation Data"),
          DT::dataTableOutput("raw_data_table")
        ),
        
        # Simulation Parameters tab
        tabPanel("Simulation Parameters",
          h3("Detailed Simulation Parameters"),
          p("This tab shows the complete simulation parameters object (my_pars) with all 24+ parameters used in the TBsim model."),
          br(),
          verbatimTextOutput("my_pars_output")
        ),
        
        # About tab
        tabPanel("About",
          h3("About TBsim"),
          p("TBsim is a tuberculosis simulation framework built on the Starsim package."),
          p("This web interface allows you to:"),
          tags$ul(
            tags$li("Configure simulation parameters"),
            tags$li("Run TB transmission simulations"),
            tags$li("Visualize results interactively"),
            tags$li("Export data for further analysis")
          ),
          br(),
          h4("Key Features"),
          tags$ul(
            tags$li("Individual-based modeling"),
            tags$li("Network-based transmission"),
            tags$li("Multiple TB states (latent, active, etc.)"),
            tags$li("Demographic processes"),
            tags$li("Intervention modeling")
          ),
          br(),
          h4("Citation"),
          p("If you use TBsim in your research, please cite the original paper and repository."),
          br(),
          h4("Links"),
          tags$a(href = "https://github.com/starsimhub/tbsim", "GitHub Repository", target = "_blank")
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive values to store simulation results
  simulation_results <- reactiveVal(NULL)
  simulation_status <- reactiveVal("Ready to run simulation")
  simulation_running <- reactiveVal(FALSE)
  my_pars <- reactiveVal(NULL)
  
  # Dark theme reactive value
  dark_theme <- reactiveVal(FALSE)
  
  # Observe dark theme toggle
  observeEvent(input$dark_theme, {
    dark_theme(input$dark_theme)
  })
  
  # Reset parameters to defaults
  observeEvent(input$reset_params, {
    updateSliderInput(session, "n_agents", value = 1000)
    updateDateInput(session, "start_date", value = as.Date("1940-01-01"))
    updateDateInput(session, "end_date", value = as.Date("2010-12-31"))
    updateSliderInput(session, "dt", value = 7)
    updateNumericInput(session, "rand_seed", value = 1)
    updateSliderInput(session, "init_prev", value = 0.05)
    updateSliderInput(session, "beta", value = 0.1)
    updateSliderInput(session, "p_latent_fast", value = 0.1)
    updateSliderInput(session, "birth_rate", value = 20)
    updateSliderInput(session, "death_rate", value = 15)
    updateSliderInput(session, "n_contacts", value = 5)
    updateSelectInput(session, "plot_scale", selected = "log")
  })
  
  # Run simulation using real TBsim model
  observeEvent(input$run_simulation, {
    simulation_running(TRUE)
    simulation_status("Running simulation...")
    
    tryCatch({
      # Set random seed
      set.seed(input$rand_seed)
      
      # Build TBsim simulation using the real model
      sim_pars <- list(
        dt = starsim$days(input$dt),
        start = as.character(input$start_date),
        stop = as.character(input$end_date),
        rand_seed = as.integer(input$rand_seed),
        verbose = 0
      )
      
      # Create population
      pop <- starsim$People(n_agents = input$n_agents)
      
      # Create TB disease model with working parameters
      tb_pars <- list(
        dt = starsim$days(input$dt),
        start = as.character(input$start_date),
        stop = as.character(input$end_date),
        init_prev = starsim$bernoulli(p = input$init_prev),
        beta = starsim$peryear(input$beta),
        p_latent_fast = starsim$bernoulli(p = input$p_latent_fast)
      )
      
      tb <- tbsim$TB(pars = tb_pars)
      
      # Create social network
      net <- starsim$RandomNet(list(
        n_contacts = starsim$poisson(lam = input$n_contacts),
        dur = 0
      ))
      
      # Create demographic processes
      births <- starsim$Births(pars = list(birth_rate = input$birth_rate))
      deaths <- starsim$Deaths(pars = list(death_rate = input$death_rate))
      
      # Create simulation
      sim <- starsim$Sim(
        people = pop,
        networks = net,
        diseases = tb,
        demographics = list(deaths, births),
        pars = sim_pars
      )
      
      # Run simulation
      sim$run()
      my_pars_value <- sim$pars
      my_pars(my_pars_value)

      # Print my_pars to console for debugging
      print(my_pars_value)

      # Extract results
      results <- sim$results$flatten()
      
      # Create time vector based on simulation parameters
      n_days <- as.numeric(input$end_date - input$start_date)
      time_days <- seq(0, n_days, by = input$dt)
      time_years <- time_days / 365.25  # Convert days to years
      
      # Store results using actual TBsim results with proper data conversion
      simulation_results(list(
        time = time_years,
        n_infected = as.numeric(results$tb_n_infected$tolist()),
        n_latent_slow = as.numeric(results$tb_n_latent_slow$tolist()),
        n_latent_fast = as.numeric(results$tb_n_latent_fast$tolist()),
        n_active = as.numeric(results$tb_n_active$tolist()),
        n_susceptible = as.numeric(results$tb_n_susceptible$tolist()),
        n_presymp = as.numeric(results$tb_n_active_presymp$tolist()),
        sim = sim,
        results = results,
        parameters = list(
          n_agents = input$n_agents,
          start_date = input$start_date,
          end_date = input$end_date,
          dt = input$dt,
          rand_seed = input$rand_seed,
          init_prev = input$init_prev,
          beta = input$beta,
          p_latent_fast = input$p_latent_fast,
          birth_rate = input$birth_rate,
          death_rate = input$death_rate,
          n_contacts = input$n_contacts
        )
      ))
      
      simulation_status("Simulation completed successfully!")
      
      simulation_running(FALSE)
      
    }, error = function(e) {
      simulation_status(paste("Error:", e$message))
      simulation_results(NULL)
      simulation_running(FALSE)
    })
  })
  
  # Status output
  output$status <- renderText({
    simulation_status()
  })
  
  # Loading spinner output
  output$loading_spinner <- renderUI({
    if (simulation_running()) {
      div(
        style = "text-align: center; padding: 20px; color: #007bff; font-size: 18px; font-weight: bold;",
        "🔄 Running simulation... Please wait"
      )
    } else {
      NULL
    }
  })
  
  # Main results plot
  output$results_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    
    # Use actual TBsim results
    if (!is.null(results$results)) {
      # Real TBsim results format
      time_data <- results$time
      infected_data <- results$n_infected
      latent_slow_data <- results$n_latent_slow
      latent_fast_data <- results$n_latent_fast
      active_data <- results$n_active
      susceptible_data <- results$n_susceptible
      presymp_data <- results$n_presymp
    } else {
      # Fallback format
      time_data <- results$time
      infected_data <- results$n_infected
      latent_slow_data <- results$n_latent_slow
      latent_fast_data <- results$n_latent_fast
      active_data <- results$n_active
      susceptible_data <- results$n_susceptible
      presymp_data <- results$n_presymp
    }
    
    # Create fancy interactive time series plot
    p <- plot_ly() %>%
    add_trace(
      x = time_data,
      y = susceptible_data,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Susceptible',
      line = list(color = '#440154', width = 3, shape = 'spline'),
      marker = list(size = 4, color = '#440154', opacity = 0.7),
      hovertemplate = '<b>Susceptible</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>',
      fill = 'tonexty',
      fillcolor = 'rgba(68, 1, 84, 0.1)'
    ) %>%
    add_trace(
      x = time_data,
      y = infected_data,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Total Infected',
      line = list(color = '#31688e', width = 3, shape = 'spline'),
      marker = list(size = 4, color = '#31688e', opacity = 0.7),
      hovertemplate = '<b>Total Infected</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>',
      fill = 'tonexty',
      fillcolor = 'rgba(49, 104, 142, 0.1)'
    ) %>%
    add_trace(
      x = time_data,
      y = latent_slow_data,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Latent Slow TB',
      line = list(color = '#35b779', width = 3, shape = 'spline'),
      marker = list(size = 4, color = '#35b779', opacity = 0.7),
      hovertemplate = '<b>Latent Slow TB</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>',
      fill = 'tonexty',
      fillcolor = 'rgba(53, 183, 121, 0.1)'
    ) %>%
    add_trace(
      x = time_data,
      y = latent_fast_data,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Latent Fast TB',
      line = list(color = '#1f9e89', width = 3, shape = 'spline'),
      marker = list(size = 4, color = '#1f9e89', opacity = 0.7),
      hovertemplate = '<b>Latent Fast TB</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>',
      fill = 'tonexty',
      fillcolor = 'rgba(31, 158, 137, 0.1)'
    ) %>%
    add_trace(
      x = time_data,
      y = presymp_data,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Pre-symptomatic',
      line = list(color = '#fde725', width = 3, shape = 'spline'),
      marker = list(size = 4, color = '#fde725', opacity = 0.7),
      hovertemplate = '<b>Pre-symptomatic</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>',
      fill = 'tonexty',
      fillcolor = 'rgba(253, 231, 37, 0.1)'
    ) %>%
    add_trace(
      x = time_data,
      y = active_data,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Active TB',
      line = list(color = '#e16462', width = 3, shape = 'spline'),
      marker = list(size = 4, color = '#e16462', opacity = 0.7),
      hovertemplate = '<b>Active TB</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>',
      fill = 'tonexty',
      fillcolor = 'rgba(225, 100, 98, 0.1)'
    ) %>%
    layout(
      title = list(
        text = "TB Simulation Results (Real TBsim Model)",
        font = list(size = 20, color = '#2c3e50'),
        x = 0.5,
        xanchor = 'center'
      ),
      xaxis = list(
        title = list(text = "Time (years)", font = list(size = 14, color = '#34495e')),
        gridcolor = 'rgba(128,128,128,0.2)',
        showgrid = TRUE,
        zeroline = FALSE,
        tickfont = list(size = 12)
      ),
      yaxis = list(
        title = list(text = "Number of Individuals", font = list(size = 14, color = '#34495e')),
        type = input$plot_scale,
        gridcolor = 'rgba(128,128,128,0.2)',
        showgrid = TRUE,
        zeroline = FALSE,
        tickfont = list(size = 12)
      ),
      hovermode = 'x unified',
      hoverlabel = list(
        bgcolor = 'rgba(255,255,255,0.9)',
        bordercolor = '#34495e',
        font = list(size = 12, color = '#2c3e50')
      ),
      legend = list(
        orientation = "v",
        x = 1.02,
        y = 1,
        bgcolor = 'rgba(255,255,255,0.8)',
        bordercolor = '#bdc3c7',
        borderwidth = 1,
        font = list(size = 12)
      ),
      plot_bgcolor = if(dark_theme()) 'rgba(26,26,26,0.8)' else 'rgba(248,249,250,0.8)',
      paper_bgcolor = if(dark_theme()) 'rgba(26,26,26,0.9)' else 'rgba(255,255,255,0.9)',
      margin = list(l = 60, r = 60, t = 80, b = 60),
      showlegend = TRUE,
      font = list(color = if(dark_theme()) '#ffffff' else '#2c3e50')
    ) %>%
    config(
      displayModeBar = TRUE,
      modeBarButtonsToRemove = c('pan2d', 'lasso2d', 'select2d'),
      displaylogo = FALSE,
      toImageButtonOptions = list(
        format = "png",
        filename = "tb_simulation",
        height = 600,
        width = 1000,
        scale = 2
      )
    )
    
    p
  })
  
  # Detailed plot
  output$detailed_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Create subplot with multiple metrics
    p1 <- plot_ly(
      x = time_data,
      y = results$n_infected,
      type = 'scatter',
      mode = 'lines',
      name = 'Total Infected'
    ) %>%
      layout(yaxis = list(title = "Count", type = "log"))
    
    p2 <- plot_ly(
      x = time_data,
      y = results$n_latent_slow,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent Slow TB'
    ) %>%
      layout(yaxis = list(title = "Count", type = "log"))
    
    p2_fast <- plot_ly(
      x = time_data,
      y = results$n_latent_fast,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent Fast TB'
    ) %>%
      layout(yaxis = list(title = "Count", type = "log"))
    
    p3 <- plot_ly(
      x = time_data,
      y = results$n_active,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB'
    ) %>%
      layout(yaxis = list(title = "Count", type = "log"))
    
    subplot(p1, p2, p2_fast, p3, nrows = 4, shareX = TRUE) %>%
      layout(
        title = "Detailed TB Simulation Metrics",
        showlegend = TRUE
      )
  })
  
  # Transitions plot
  output$transitions_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Calculate transition rates if available
    if ("tb_new_infections" %in% names(results$results)) {
      p <- plot_ly(
        x = time_data,
        y = as.numeric(results$results$tb_new_infections$tolist()),
        type = 'scatter',
        mode = 'lines',
        name = 'New Infections'
      ) %>%
        layout(
          title = "TB State Transitions",
          xaxis = list(title = "Time"),
          yaxis = list(title = "New Cases per Time Step", type = "log")
        )
    } else {
      # Fallback plot
      p <- plot_ly(
        x = time_data,
        y = results$n_infected,
        type = 'scatter',
        mode = 'lines',
        name = 'Total Infected'
      ) %>%
        layout(
          title = "TB State Transitions",
          xaxis = list(title = "Time"),
          yaxis = list(title = "Number of Individuals", type = "log")
        )
    }
    
    p
  })
  
  # Summary table
  output$summary_table <- DT::renderDataTable({
    req(simulation_results())
    
    results <- simulation_results()$results
    params <- simulation_results()$parameters
    
    # Calculate summary statistics
    summary_data <- data.frame(
      Metric = c(
        "Population Size",
        "Simulation Duration (years)",
        "Initial Prevalence",
        "Transmission Rate",
        "Final Infected Count",
        "Peak Infected Count",
        "Final Latent Slow Count",
        "Final Latent Fast Count",
        "Final Active Count"
      ),
      Value = c(
        params$n_agents,
        round((params$end_date - params$start_date) / 365.25, 2),
        params$init_prev,
        params$beta,
        max(as.numeric(results$tb_n_infected$tolist()), na.rm = TRUE),
        max(as.numeric(results$tb_n_infected$tolist()), na.rm = TRUE),
        max(as.numeric(results$tb_n_latent_slow$tolist()), na.rm = TRUE),
        max(as.numeric(results$tb_n_latent_fast$tolist()), na.rm = TRUE),
        max(as.numeric(results$tb_n_active$tolist()), na.rm = TRUE)
      )
    )
    
    DT::datatable(
      summary_data,
      options = list(
        pageLength = 10,
        searching = FALSE,
        ordering = FALSE
      ),
      rownames = FALSE
    )
  })
  
  # Parameters table
  output$parameters_table <- DT::renderDataTable({
    req(simulation_results())
    
    results <- simulation_results()
    params <- results$parameters
    
    # Create comprehensive parameters table
    parameters_data <- data.frame(
      Category = c(
        "Basic Settings",
        "Basic Settings", 
        "Basic Settings",
        "Basic Settings",
        "Basic Settings",
        "TB Disease Parameters",
        "TB Disease Parameters",
        "TB Disease Parameters",
        "TB State Transition Rates",
        "TB State Transition Rates",
        "TB State Transition Rates",
        "TB State Transition Rates",
        "TB State Transition Rates",
        "TB Mortality Rates",
        "TB Mortality Rates",
        "TB Mortality Rates",
        "TB Transmissibility",
        "TB Transmissibility",
        "TB Transmissibility",
        "TB Transmissibility",
        "TB Transmissibility",
        "TB Susceptibility",
        "Demographics",
        "Demographics",
        "Social Network"
      ),
      Parameter = c(
        "Population Size",
        "Start Date",
        "End Date", 
        "Time Step (days)",
        "Random Seed",
        "Initial Prevalence",
        "Transmission Rate (per year)",
        "Probability of Fast Latent TB",
        "Latent Slow → Pre-symptomatic (per day)",
        "Latent Fast → Pre-symptomatic (per day)",
        "Pre-symptomatic → Active (per day)",
        "Active → Clearance (per day)",
        "Treatment → Clearance (per year)",
        "Extra-Pulmonary TB → Death (per day)",
        "Smear Positive → Death (per day)",
        "Smear Negative → Death (per day)",
        "Pre-symptomatic Relative Transmissibility",
        "Smear Positive Relative Transmissibility",
        "Smear Negative Relative Transmissibility",
        "Extra-Pulmonary Relative Transmissibility",
        "Treatment Effect on Transmissibility",
        "Latent Slow Relative Susceptibility",
        "Birth Rate (per 1000)",
        "Death Rate (per 1000)",
        "Average Contacts per Person"
      ),
      Value = c(
        params$n_agents,
        as.character(params$start_date),
        as.character(params$end_date),
        params$dt,
        params$rand_seed,
        params$init_prev,
        params$beta,
        params$p_latent_fast,
        input$rate_LS_to_presym,
        input$rate_LF_to_presym,
        input$rate_presym_to_active,
        input$rate_active_to_clear,
        input$rate_treatment_to_clear,
        input$rate_exptb_to_dead,
        input$rate_smpos_to_dead,
        input$rate_smneg_to_dead,
        input$rel_trans_presymp,
        input$rel_trans_smpos,
        input$rel_trans_smneg,
        input$rel_trans_exptb,
        input$rel_trans_treatment,
        input$rel_sus_latentslow,
        input$birth_rate,
        input$death_rate,
        input$n_contacts
      )
    )
    
    DT::datatable(
      parameters_data,
      options = list(
        pageLength = 15,
        scrollY = "400px",
        searching = TRUE,
        ordering = TRUE
      ),
      rownames = FALSE,
      filter = 'top'
    )
  })
  
  # Raw simulation parameters output
  output$my_pars_output <- renderText({
    if (is.null(my_pars())) {
      return("No simulation has been run yet. Please run a simulation first to see the parameters.")
    }
    
    tryCatch({
      # Use Python's built-in dict() conversion to avoid sciris issues
      pars_dict <- my_pars()$dict()
      pars_list <- py_to_r(pars_dict)
      
      output_lines <- c()
      output_lines <- c(output_lines, "=== SIMULATION PARAMETERS (my_pars) ===")
      output_lines <- c(output_lines, paste("Object type:", class(my_pars())))
      output_lines <- c(output_lines, paste("Number of parameters:", length(pars_list)))
      output_lines <- c(output_lines, "")
      output_lines <- c(output_lines, "Detailed Parameters:")
      output_lines <- c(output_lines, "")
      
      # Show each parameter with its value - use safe iteration
      param_names <- names(pars_list)
      if (is.null(param_names)) {
        # If no names, use indices
        for (i in seq_along(pars_list)) {
          param_name <- paste("Parameter", i)
          param_value <- pars_list[[i]]
          value_str <- toString(param_value)
          output_lines <- c(output_lines, paste(param_name, ":", value_str))
        }
      } else {
        # Use names if available
        for (i in seq_along(pars_list)) {
          param_name <- param_names[i]
          if (is.na(param_name) || param_name == "") {
            param_name <- paste("Parameter", i)
          }
          param_value <- pars_list[[i]]
          value_str <- toString(param_value)
          output_lines <- c(output_lines, paste(param_name, ":", value_str))
        }
      }
      
      paste(output_lines, collapse = "\n")
    }, error = function(e) {
      # Fallback: try simple string representation
      tryCatch({
        simple_repr <- my_pars()$`__str__`()
        paste("=== SIMULATION PARAMETERS (my_pars) ===\n",
              "Object type:", class(my_pars()), "\n",
              "String representation:\n", 
              py_to_r(simple_repr))
      }, error = function(e2) {
        paste("Error accessing my_pars:", e$message, "\n",
              "Fallback error:", e2$message, "\n",
              "Object type:", class(my_pars()), "\n",
              "Object length:", length(my_pars()))
      })
    })
  })
  
  # Raw data table
  output$raw_data_table <- DT::renderDataTable({
    req(simulation_results())
    
    results <- simulation_results()
    
    # Use the calculated time vector from simulation_results
    time_data <- results$time
    
    # Convert results to data frame
    df <- data.frame(
      Time = time_data,
      Infected = results$n_infected,
      Latent_Slow = results$n_latent_slow,
      Latent_Fast = results$n_latent_fast,
      Active = results$n_active
    )
    
    DT::datatable(
      df,
      options = list(
        pageLength = 20,
        scrollX = TRUE
      )
    )
  })
  
  # Advanced Visualizations
  
  # Disease Prevalence Plot
  output$prevalence_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Calculate prevalence rates
    total_pop <- results$n_susceptible + results$n_infected
    prevalence_infected <- (results$n_infected / total_pop) * 100
    prevalence_active <- (results$n_active / total_pop) * 100
    prevalence_latent_slow <- (results$n_latent_slow / total_pop) * 100
    prevalence_latent_fast <- (results$n_latent_fast / total_pop) * 100
    
    p <- plot_ly() %>%
    add_trace(
      x = time_data,
      y = prevalence_infected,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Total Infected %',
      line = list(color = '#31688e', width = 3),
      marker = list(size = 4, color = '#31688e'),
      hovertemplate = '<b>Total Infected</b><br>Time: %{x:.2f} years<br>Prevalence: %{y:.2f}%<extra></extra>'
    ) %>%
    add_trace(
      x = time_data,
      y = prevalence_active,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Active TB %',
      line = list(color = '#e16462', width = 3),
      marker = list(size = 4, color = '#e16462'),
      hovertemplate = '<b>Active TB</b><br>Time: %{x:.2f} years<br>Prevalence: %{y:.2f}%<extra></extra>'
    ) %>%
    add_trace(
      x = time_data,
      y = prevalence_latent_slow,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Latent Slow TB %',
      line = list(color = '#35b779', width = 3),
      marker = list(size = 4, color = '#35b779'),
      hovertemplate = '<b>Latent Slow TB</b><br>Time: %{x:.2f} years<br>Prevalence: %{y:.2f}%<extra></extra>'
    ) %>%
    add_trace(
      x = time_data,
      y = prevalence_latent_fast,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Latent Fast TB %',
      line = list(color = '#1f9e89', width = 3),
      marker = list(size = 4, color = '#1f9e89'),
      hovertemplate = '<b>Latent Fast TB</b><br>Time: %{x:.2f} years<br>Prevalence: %{y:.2f}%<extra></extra>'
    ) %>%
    layout(
      title = "TB Disease Prevalence Over Time",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Prevalence (%)"),
      hovermode = 'x unified',
      legend = list(orientation = "v", x = 1.02, y = 1)
    )
    
    p
  })
  
  # Enhanced Transmission Sankey Diagram
  output$transmission_sankey <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    
    # Calculate actual flow rates from simulation data
    n_points <- length(results$time)
    if (n_points > 1) {
      # Calculate transitions between states using actual simulation parameters
      p_latent_fast <- results$parameters$p_latent_fast
      p_latent_slow <- 1 - p_latent_fast
      
      # Calculate flow rates with better statistical methods
      susceptible_to_latent_slow <- abs(diff(results$n_susceptible)) * p_latent_slow
      susceptible_to_latent_fast <- abs(diff(results$n_susceptible)) * p_latent_fast
      latent_slow_to_presymp <- abs(diff(results$n_latent_slow))
      latent_fast_to_presymp <- abs(diff(results$n_latent_fast))
      presymp_to_active <- abs(diff(results$n_presymp))
      active_to_clear <- abs(diff(results$n_active))
      
      # Calculate death flows from actual death data
      if (!is.null(results$n_deaths)) {
        death_flow <- mean(results$n_deaths, na.rm = TRUE)
      } else {
        death_flow <- 0
      }
      
      # Calculate flow rates based on actual simulation population
      # Use real population values, not scaled to millions
      flow_values <- c(
        max(mean(susceptible_to_latent_slow, na.rm = TRUE), 0.1),
        max(mean(susceptible_to_latent_fast, na.rm = TRUE), 0.1),
        max(mean(latent_slow_to_presymp, na.rm = TRUE), 0.1),
        max(mean(latent_fast_to_presymp, na.rm = TRUE), 0.1),
        max(mean(presymp_to_active, na.rm = TRUE), 0.1),
        max(mean(active_to_clear, na.rm = TRUE), 0.1),
        max(death_flow, 0.1)
      )
      
      # Get actual population size from simulation parameters
      actual_population <- results$parameters$n_agents
      
      # Enhanced Sankey diagram with premium styling
      p <- plot_ly(
        type = "sankey",
        orientation = "h",
        arrangement = "snap",
        node = list(
          label = c("Susceptible", "Latent Slow", "Latent Fast", "Pre-symptomatic", "Active", "Cleared", "Death"),
          color = c(
            'rgba(68, 1, 84, 0.9)',      # Deep purple for Susceptible
            'rgba(53, 183, 121, 0.9)',   # Green for Latent Slow
            'rgba(31, 158, 137, 0.9)',   # Teal for Latent Fast
            'rgba(253, 231, 37, 0.9)',   # Yellow for Pre-symptomatic
            'rgba(225, 100, 98, 0.9)',   # Red for Active
            'rgba(49, 104, 142, 0.9)',   # Blue for Cleared
            'rgba(142, 68, 173, 0.9)'    # Purple for Death
          ),
          pad = 25,
          thickness = 30,
          line = list(
            color = "rgba(255, 255, 255, 0.8)",
            width = 2
          ),
          hovertemplate = paste0('<b>%{label}</b><br>Population: %{value:,.0f} individuals<br>Total Population: ', actual_population, '<br><extra></extra>')
        ),
        link = list(
          source = c(0, 0, 1, 2, 3, 4, 4),
          target = c(1, 2, 3, 3, 4, 5, 6),
          value = flow_values,
          color = c(
            'rgba(53, 183, 121, 0.7)',   # Green flow
            'rgba(31, 158, 137, 0.7)',   # Teal flow
            'rgba(253, 231, 37, 0.7)',   # Yellow flow
            'rgba(253, 231, 37, 0.7)',   # Yellow flow
            'rgba(225, 100, 98, 0.7)',   # Red flow
            'rgba(49, 104, 142, 0.7)',   # Blue flow
            'rgba(142, 68, 173, 0.7)'    # Purple flow
          ),
          hovertemplate = paste0('<b>%{source.label} → %{target.label}</b><br>Flow: %{value:,.0f} individuals<br>Total Population: ', actual_population, '<br><extra></extra>')
        )
      ) %>%
      layout(
        title = list(
          text = paste0("🔄 TB Disease Progression Flow - Population: ", format(actual_population, big.mark = ","), " individuals"),
          font = list(
            size = 20,
            color = if(dark_theme()) '#ffffff' else '#2c3e50',
            family = "Arial, sans-serif"
          ),
          x = 0.5,
          xanchor = 'center'
        ),
        font = list(
          size = 14,
          color = if(dark_theme()) '#e8e8e8' else '#2c3e50',
          family = "Arial, sans-serif"
        ),
        margin = list(l = 80, r = 80, t = 80, b = 80),
        plot_bgcolor = if(dark_theme()) 'rgba(15, 15, 35, 0.8)' else 'rgba(248, 249, 250, 0.8)',
        paper_bgcolor = if(dark_theme()) 'rgba(15, 15, 35, 0.9)' else 'rgba(255, 255, 255, 0.9)',
        annotations = list(
          list(
            x = 0.5, y = -0.1,
            xref = "paper", yref = "paper",
            text = paste0("💡 Hover over nodes and links for detailed information | Simulation Population: ", format(actual_population, big.mark = ","), " individuals"),
            showarrow = FALSE,
            font = list(
              size = 12,
              color = if(dark_theme()) '#b8b8c8' else '#6c757d'
            )
          )
        )
      ) %>%
      config(
        displayModeBar = TRUE,
        modeBarButtonsToRemove = c('pan2d', 'lasso2d', 'select2d', 'autoScale2d'),
        displaylogo = FALSE,
        toImageButtonOptions = list(
          format = "png",
          filename = "tb_disease_flow",
          height = 800,
          width = 1200,
          scale = 2
        )
      )
    } else {
      # Enhanced fallback for insufficient data
      actual_population <- results$parameters$n_agents
      p <- plot_ly() %>%
      add_annotation(
        text = "🚀 Run a simulation to see the enhanced transmission flow visualization",
        x = 0.5, y = 0.5,
        xref = "paper", yref = "paper",
        showarrow = FALSE,
        font = list(
          size = 18,
          color = if(dark_theme()) '#e8e8e8' else '#2c3e50',
          family = "Arial, sans-serif"
        )
      ) %>%
      add_annotation(
        text = paste0("This interactive Sankey diagram will show disease progression flows for population of ", format(actual_population, big.mark = ","), " individuals"),
        x = 0.5, y = 0.4,
        xref = "paper", yref = "paper",
        showarrow = FALSE,
        font = list(
          size = 14,
          color = if(dark_theme()) '#b8b8c8' else '#6c757d',
          family = "Arial, sans-serif"
        )
      ) %>%
      layout(
        title = list(
          text = paste0("🔄 TB Disease Progression Flow - Population: ", format(actual_population, big.mark = ","), " individuals"),
          font = list(
            size = 20,
            color = if(dark_theme()) '#ffffff' else '#2c3e50'
          )
        ),
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        plot_bgcolor = if(dark_theme()) 'rgba(15, 15, 35, 0.8)' else 'rgba(248, 249, 250, 0.8)',
        paper_bgcolor = if(dark_theme()) 'rgba(15, 15, 35, 0.9)' else 'rgba(255, 255, 255, 0.9)'
      )
    }
    
    p
  })
  
  # Interactive Time Series
  output$interactive_timeseries <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Create interactive subplot
    p1 <- plot_ly(
      x = time_data,
      y = results$n_susceptible,
      type = 'scatter',
      mode = 'lines',
      name = 'Susceptible',
      line = list(color = '#440154', width = 2),
      hovertemplate = '<b>Susceptible</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>'
    ) %>%
    layout(yaxis = list(title = "Susceptible", type = "log"))
    
    p2 <- plot_ly(
      x = time_data,
      y = results$n_infected,
      type = 'scatter',
      mode = 'lines',
      name = 'Total Infected',
      line = list(color = '#31688e', width = 2),
      hovertemplate = '<b>Total Infected</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>'
    ) %>%
    layout(yaxis = list(title = "Total Infected", type = "log"))
    
    p3 <- plot_ly(
      x = time_data,
      y = results$n_active,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB',
      line = list(color = '#e16462', width = 2),
      hovertemplate = '<b>Active TB</b><br>Time: %{x:.2f} years<br>Count: %{y:,.0f}<extra></extra>'
    ) %>%
    layout(yaxis = list(title = "Active TB", type = "log"))
    
    subplot(p1, p2, p3, nrows = 3, shareX = TRUE) %>%
    layout(
      title = "Interactive TB Simulation Time Series",
      showlegend = FALSE,
      margin = list(l = 60, r = 30, t = 50, b = 50)
    )
  })
  
  # Enhanced Disease Progression
  output$disease_progression <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Enhanced percent-stacked area chart with premium styling
    p <- plot_ly(
      x = time_data,
      y = results$n_susceptible,
      type = 'scatter',
      mode = 'lines',
      stackgroup = 'population',
      groupnorm = 'percent',
      name = '🟣 Susceptible',
      line = list(color = '#440154', width = 2, shape = 'spline'),
      fillcolor = 'rgba(68, 1, 84, 0.6)',
      hovertemplate = '<b>🟣 Susceptible</b><br>Time: %{x:.2f} years<br>Percentage: %{y:.2f}%<br>Count: %{customdata:,.0f}<extra></extra>',
      customdata = results$n_susceptible
    ) %>%
    add_trace(
      x = time_data,
      y = results$n_latent_slow,
      type = 'scatter',
      mode = 'lines',
      stackgroup = 'population',
      name = '🟢 Latent Slow',
      line = list(color = '#35b779', width = 2, shape = 'spline'),
      fillcolor = 'rgba(53, 183, 121, 0.6)',
      hovertemplate = '<b>🟢 Latent Slow</b><br>Time: %{x:.2f} years<br>Percentage: %{y:.2f}%<br>Count: %{customdata:,.0f}<extra></extra>',
      customdata = results$n_latent_slow
    ) %>%
    add_trace(
      x = time_data,
      y = results$n_latent_fast,
      type = 'scatter',
      mode = 'lines',
      stackgroup = 'population',
      name = '🔵 Latent Fast',
      line = list(color = '#1f9e89', width = 2, shape = 'spline'),
      fillcolor = 'rgba(31, 158, 137, 0.6)',
      hovertemplate = '<b>🔵 Latent Fast</b><br>Time: %{x:.2f} years<br>Percentage: %{y:.2f}%<br>Count: %{customdata:,.0f}<extra></extra>',
      customdata = results$n_latent_fast
    ) %>%
    add_trace(
      x = time_data,
      y = results$n_presymp,
      type = 'scatter',
      mode = 'lines',
      stackgroup = 'population',
      name = '🟡 Pre-symptomatic',
      line = list(color = '#fde725', width = 2, shape = 'spline'),
      fillcolor = 'rgba(253, 231, 37, 0.6)',
      hovertemplate = '<b>🟡 Pre-symptomatic</b><br>Time: %{x:.2f} years<br>Percentage: %{y:.2f}%<br>Count: %{customdata:,.0f}<extra></extra>',
      customdata = results$n_presymp
    ) %>%
    add_trace(
      x = time_data,
      y = results$n_active,
      type = 'scatter',
      mode = 'lines',
      stackgroup = 'population',
      name = '🔴 Active',
      line = list(color = '#e16462', width = 2, shape = 'spline'),
      fillcolor = 'rgba(225, 100, 98, 0.6)',
      hovertemplate = '<b>🔴 Active</b><br>Time: %{x:.2f} years<br>Percentage: %{y:.2f}%<br>Count: %{customdata:,.0f}<extra></extra>',
      customdata = results$n_active
    ) %>%
    layout(
      title = list(
        text = "📊 Disease Progression - Population Distribution Over Time",
        font = list(
          size = 18,
          color = if(dark_theme()) '#ffffff' else '#2c3e50',
          family = "Arial, sans-serif"
        ),
        x = 0.5,
        xanchor = 'center'
      ),
      xaxis = list(
        title = list(
          text = "Time (years)",
          font = list(size = 14, color = if(dark_theme()) '#e8e8e8' else '#2c3e50')
        ),
        gridcolor = if(dark_theme()) 'rgba(255,255,255,0.1)' else 'rgba(0,0,0,0.1)',
        showgrid = TRUE,
        zeroline = FALSE,
        tickfont = list(size = 12, color = if(dark_theme()) '#b8b8c8' else '#6c757d')
      ),
      yaxis = list(
        title = list(
          text = "Population (%)",
          font = list(size = 14, color = if(dark_theme()) '#e8e8e8' else '#2c3e50')
        ),
        range = c(0, 100),
        gridcolor = if(dark_theme()) 'rgba(255,255,255,0.1)' else 'rgba(0,0,0,0.1)',
        showgrid = TRUE,
        zeroline = FALSE,
        tickfont = list(size = 12, color = if(dark_theme()) '#b8b8c8' else '#6c757d')
      ),
      hovermode = 'x unified',
      hoverlabel = list(
        bgcolor = if(dark_theme()) 'rgba(30, 30, 46, 0.9)' else 'rgba(255,255,255,0.9)',
        bordercolor = if(dark_theme()) '#3a3a4a' else '#dee2e6',
        font = list(size = 12, color = if(dark_theme()) '#e8e8e8' else '#2c3e50')
      ),
      legend = list(
        orientation = "v",
        x = 1.02,
        y = 1,
        bgcolor = if(dark_theme()) 'rgba(30, 30, 46, 0.8)' else 'rgba(255,255,255,0.8)',
        bordercolor = if(dark_theme()) '#3a3a4a' else '#dee2e6',
        borderwidth = 1,
        font = list(size = 12, color = if(dark_theme()) '#e8e8e8' else '#2c3e50')
      ),
      plot_bgcolor = if(dark_theme()) 'rgba(15, 15, 35, 0.8)' else 'rgba(248,249,250,0.8)',
      paper_bgcolor = if(dark_theme()) 'rgba(15, 15, 35, 0.9)' else 'rgba(255,255,255,0.9)',
      margin = list(l = 60, r = 60, t = 80, b = 60),
      showlegend = TRUE,
      font = list(color = if(dark_theme()) '#e8e8e8' else '#2c3e50'),
      annotations = list(
        list(
          x = 0.5, y = -0.15,
          xref = "paper", yref = "paper",
          text = "💡 Hover over the chart to see detailed population percentages and counts",
          showarrow = FALSE,
          font = list(
            size = 12,
            color = if(dark_theme()) '#b8b8c8' else '#6c757d'
          )
        )
      )
    ) %>%
    config(
      displayModeBar = TRUE,
      modeBarButtonsToRemove = c('pan2d', 'lasso2d', 'select2d'),
      displaylogo = FALSE,
      toImageButtonOptions = list(
        format = "png",
        filename = "tb_disease_progression",
        height = 600,
        width = 1000,
        scale = 2
      )
    )
    
    p
  })
  
  # Mortality Analysis
  output$mortality_analysis <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Calculate mortality rate over time
    if (length(results$n_active) > 1) {
      mortality_rate <- diff(results$n_active) / results$n_active[-1] * 100
      mortality_rate <- c(0, mortality_rate) # Add initial value
    } else {
      mortality_rate <- rep(0, length(time_data))
    }
    
    p <- plot_ly(
      x = time_data,
      y = mortality_rate,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Mortality Rate',
      line = list(color = '#e74c3c', width = 3),
      marker = list(size = 6, color = '#e74c3c'),
      hovertemplate = '<b>Mortality Rate</b><br>Time: %{x:.2f} years<br>Rate: %{y:.2f}%<extra></extra>'
    ) %>%
    layout(
      title = "TB Mortality Rate Over Time",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Mortality Rate (%)"),
      shapes = list(
        list(
          type = "line",
          x0 = min(time_data), x1 = max(time_data),
          y0 = mean(mortality_rate, na.rm = TRUE), y1 = mean(mortality_rate, na.rm = TRUE),
          line = list(color = "red", width = 2, dash = "dash")
        )
      ),
      annotations = list(
        list(
          x = max(time_data) * 0.7,
          y = mean(mortality_rate, na.rm = TRUE) * 1.1,
          text = paste("Average:", round(mean(mortality_rate, na.rm = TRUE), 2), "%"),
          showarrow = FALSE,
          font = list(size = 12, color = "red")
        )
      )
    )
    
    p
  })
  
  # TB Death Metrics
  
  # Death trends over time
  output$death_trends_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Extract death metrics from results using correct key names
    if ("tb_new_deaths" %in% names(results$results)) {
      new_deaths <- as.numeric(results$results$tb_new_deaths$tolist())
      new_deaths_15plus <- as.numeric(results$results$`tb_new_deaths_15+`$tolist())
    } else {
      # Fallback if death data not available
      new_deaths <- rep(0, length(time_data))
      new_deaths_15plus <- rep(0, length(time_data))
    }
    
    p <- plot_ly() %>%
      add_trace(
        x = time_data,
        y = new_deaths,
        type = 'scatter',
        mode = 'lines+markers',
        name = 'All Ages',
        line = list(color = '#e74c3c', width = 3),
        marker = list(size = 4, color = '#e74c3c'),
        hovertemplate = '<b>All Ages Deaths</b><br>Time: %{x:.2f} years<br>Deaths: %{y:,.0f}<extra></extra>'
      ) %>%
      add_trace(
        x = time_data,
        y = new_deaths_15plus,
        type = 'scatter',
        mode = 'lines+markers',
        name = '15+ Years',
        line = list(color = '#c0392b', width = 3),
        marker = list(size = 4, color = '#c0392b'),
        hovertemplate = '<b>15+ Years Deaths</b><br>Time: %{x:.2f} years<br>Deaths: %{y:,.0f}<extra></extra>'
      ) %>%
      layout(
        title = "TB Deaths Over Time",
        xaxis = list(title = "Time (years)"),
        yaxis = list(title = "Number of Deaths", type = "log"),
        hovermode = 'x unified',
        legend = list(orientation = "v", x = 1.02, y = 1)
      )
    
    p
  })
  
  # Cumulative deaths plot
  output$cumulative_deaths_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Extract cumulative death metrics using correct key names
    if ("tb_cum_deaths" %in% names(results$results)) {
      cum_deaths <- as.numeric(results$results$tb_cum_deaths$tolist())
      cum_deaths_15plus <- as.numeric(results$results$`tb_cum_deaths_15+`$tolist())
    } else {
      # Fallback if death data not available
      cum_deaths <- rep(0, length(time_data))
      cum_deaths_15plus <- rep(0, length(time_data))
    }
    
    p <- plot_ly() %>%
      add_trace(
        x = time_data,
        y = cum_deaths,
        type = 'scatter',
        mode = 'lines',
        name = 'All Ages (Cumulative)',
        line = list(color = '#8e44ad', width = 3),
        fill = 'tonexty',
        fillcolor = 'rgba(142, 68, 173, 0.1)',
        hovertemplate = '<b>All Ages Cumulative</b><br>Time: %{x:.2f} years<br>Total Deaths: %{y:,.0f}<extra></extra>'
      ) %>%
      add_trace(
        x = time_data,
        y = cum_deaths_15plus,
        type = 'scatter',
        mode = 'lines',
        name = '15+ Years (Cumulative)',
        line = list(color = '#9b59b6', width = 3),
        fill = 'tonexty',
        fillcolor = 'rgba(155, 89, 182, 0.1)',
        hovertemplate = '<b>15+ Years Cumulative</b><br>Time: %{x:.2f} years<br>Total Deaths: %{y:,.0f}<extra></extra>'
      ) %>%
      layout(
        title = "Cumulative TB Deaths",
        xaxis = list(title = "Time (years)"),
        yaxis = list(title = "Cumulative Deaths"),
        hovermode = 'x unified',
        legend = list(orientation = "v", x = 1.02, y = 1)
      )
    
    p
  })
  
  # Death summary statistics table
  output$death_summary_table <- DT::renderDataTable({
    req(simulation_results())
    
    results <- simulation_results()
    
    # Extract death metrics using correct key names
    if ("tb_new_deaths" %in% names(results$results)) {
      new_deaths <- as.numeric(results$results$tb_new_deaths$tolist())
      new_deaths_15plus <- as.numeric(results$results$`tb_new_deaths_15+`$tolist())
      cum_deaths <- as.numeric(results$results$tb_cum_deaths$tolist())
      cum_deaths_15plus <- as.numeric(results$results$`tb_cum_deaths_15+`$tolist())
      deaths_ppy <- as.numeric(results$results$tb_deaths_ppy$tolist())
    } else {
      # Fallback if death data not available
      new_deaths <- rep(0, length(results$time))
      new_deaths_15plus <- rep(0, length(results$time))
      cum_deaths <- rep(0, length(results$time))
      cum_deaths_15plus <- rep(0, length(results$time))
      deaths_ppy <- rep(0, length(results$time))
    }
    
    # Calculate summary statistics
    total_deaths <- max(cum_deaths, na.rm = TRUE)
    total_deaths_15plus <- max(cum_deaths_15plus, na.rm = TRUE)
    peak_daily_deaths <- max(new_deaths, na.rm = TRUE)
    peak_daily_deaths_15plus <- max(new_deaths_15plus, na.rm = TRUE)
    avg_death_rate <- mean(deaths_ppy, na.rm = TRUE)
    max_death_rate <- max(deaths_ppy, na.rm = TRUE)
    
    # Create summary table
    death_summary <- data.frame(
      Metric = c(
        "Total TB Deaths (All Ages)",
        "Total TB Deaths (15+ Years)",
        "Peak Daily Deaths (All Ages)",
        "Peak Daily Deaths (15+ Years)",
        "Average Death Rate (per person-year)",
        "Maximum Death Rate (per person-year)",
        "Death Rate (15+ Years as % of Total)",
        "Simulation Duration (years)"
      ),
      Value = c(
        format(total_deaths, big.mark = ","),
        format(total_deaths_15plus, big.mark = ","),
        format(peak_daily_deaths, big.mark = ","),
        format(peak_daily_deaths_15plus, big.mark = ","),
        sprintf("%.6f", avg_death_rate),
        sprintf("%.6f", max_death_rate),
        sprintf("%.1f%%", if(total_deaths > 0) (total_deaths_15plus / total_deaths * 100) else 0),
        sprintf("%.1f", max(results$time, na.rm = TRUE))
      )
    )
    
    DT::datatable(
      death_summary,
      options = list(
        pageLength = 10,
        searching = FALSE,
        ordering = FALSE
      ),
      rownames = FALSE
    )
  })
  
  # Age-specific deaths plot
  output$age_specific_deaths_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Extract age-specific death data using correct key names
    if ("tb_new_deaths" %in% names(results$results)) {
      new_deaths_all <- as.numeric(results$results$tb_new_deaths$tolist())
      new_deaths_15plus <- as.numeric(results$results$`tb_new_deaths_15+`$tolist())
      new_deaths_under15 <- new_deaths_all - new_deaths_15plus
    } else {
      # Fallback if death data not available
      new_deaths_all <- rep(0, length(time_data))
      new_deaths_15plus <- rep(0, length(time_data))
      new_deaths_under15 <- rep(0, length(time_data))
    }
    
    # Create stacked area chart
    p <- plot_ly(
      x = time_data,
      y = new_deaths_under15,
      type = 'scatter',
      mode = 'lines',
      fill = 'tonexty',
      name = 'Under 15 Years',
      line = list(color = '#f39c12', width = 0),
      fillcolor = 'rgba(243, 156, 18, 0.6)',
      hovertemplate = '<b>Under 15 Years</b><br>Time: %{x:.2f} years<br>Deaths: %{y:,.0f}<extra></extra>'
    ) %>%
    add_trace(
      x = time_data,
      y = new_deaths_15plus,
      type = 'scatter',
      mode = 'lines',
      fill = 'tonexty',
      name = '15+ Years',
      line = list(color = '#e67e22', width = 0),
      fillcolor = 'rgba(230, 126, 34, 0.6)',
      hovertemplate = '<b>15+ Years</b><br>Time: %{x:.2f} years<br>Deaths: %{y:,.0f}<extra></extra>'
    ) %>%
    layout(
      title = "Age-Specific TB Deaths",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Number of Deaths"),
      hovermode = 'x unified',
      legend = list(orientation = "v", x = 1.02, y = 1)
    )
    
    p
  })
  
  # Death rate analysis plot
  output$death_rate_analysis_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    time_data <- results$time
    
    # Extract death rate data
    if ("tb_deaths_ppy" %in% names(results$results)) {
      deaths_ppy <- as.numeric(results$results$tb_deaths_ppy$tolist())
    } else {
      # Fallback if death rate data not available
      deaths_ppy <- rep(0, length(time_data))
    }
    
    p <- plot_ly(
      x = time_data,
      y = deaths_ppy,
      type = 'scatter',
      mode = 'lines+markers',
      name = 'Death Rate',
      line = list(color = '#e74c3c', width = 3),
      marker = list(size = 4, color = '#e74c3c'),
      hovertemplate = '<b>Death Rate</b><br>Time: %{x:.2f} years<br>Rate: %{y:.6f} per person-year<extra></extra>'
    ) %>%
    layout(
      title = "TB Death Rate Over Time",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Death Rate (per person-year)", type = "log"),
      shapes = list(
        list(
          type = "line",
          x0 = min(time_data), x1 = max(time_data),
          y0 = mean(deaths_ppy, na.rm = TRUE), y1 = mean(deaths_ppy, na.rm = TRUE),
          line = list(color = "red", width = 2, dash = "dash")
        )
      ),
      annotations = list(
        list(
          x = max(time_data) * 0.7,
          y = mean(deaths_ppy, na.rm = TRUE) * 1.1,
          text = paste("Average:", sprintf("%.6f", mean(deaths_ppy, na.rm = TRUE))),
          showarrow = FALSE,
          font = list(size = 12, color = "red")
        )
      )
    )
    
    p
  })
}

# Run the application
shinyApp(ui = ui, server = server)
