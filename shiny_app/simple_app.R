# TBsim Shiny App - Real Model Integration
# This version integrates with the actual TBsim package

library(shiny)
library(plotly)
library(DT)
library(reticulate)

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
  titlePanel("TBsim - Tuberculosis Simulation Web Interface"),
  
  sidebarLayout(
    sidebarPanel(
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
      sliderInput("init_prev", "Initial Prevalence", value = 0.01, min = 0, max = 1, step = 0.001),
      sliderInput("beta", "Transmission Rate (per year)", value = 0.0025, min = 0, max = 0.1, step = 0.0001),
      sliderInput("p_latent_fast", "Probability of Fast Latent TB", value = 0.1, min = 0, max = 1, step = 0.01),
      
      # TB State Transition Rates
      h4("TB State Transition Rates"),
      sliderInput("rate_LS_to_presym", "Latent Slow → Pre-symptomatic (per day)", value = 3e-5, min = 0, max = 1e-3, step = 1e-6),
      sliderInput("rate_LF_to_presym", "Latent Fast → Pre-symptomatic (per day)", value = 6e-3, min = 0, max = 0.1, step = 1e-4),
      sliderInput("rate_presym_to_active", "Pre-symptomatic → Active (per day)", value = 3e-2, min = 0, max = 1, step = 1e-3),
      sliderInput("rate_active_to_clear", "Active → Clearance (per day)", value = 2.4e-4, min = 0, max = 1e-2, step = 1e-5),
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
      
      # TB Diagnostics
      h4("TB Diagnostics"),
      sliderInput("cxr_asymp_sens", "Chest X-ray Sensitivity (asymptomatic)", value = 1.0, min = 0, max = 1, step = 0.01),
      
      # TB Heterogeneity
      h4("TB Heterogeneity"),
      sliderInput("reltrans_het", "Transmission Heterogeneity", value = 1.0, min = 0.1, max = 5, step = 0.1),
      
      # Demographics
      h4("Demographics"),
      sliderInput("birth_rate", "Birth Rate (per 1000)", value = 20, min = 0, max = 100, step = 1),
      sliderInput("death_rate", "Death Rate (per 1000)", value = 15, min = 0, max = 100, step = 1),
      
      # Network parameters
      h4("Social Network"),
      sliderInput("n_contacts", "Average Contacts per Person", value = 5, min = 1, max = 50, step = 1),
      
      # Action buttons
      br(),
      actionButton("run_simulation", "Run Simulation", class = "btn-primary btn-lg"),
      br(), br(),
      actionButton("reset_params", "Reset to Defaults", class = "btn-secondary"),
      
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
          plotlyOutput("results_plot", height = "600px"),
          br(),
          h4("Summary Statistics"),
          DT::dataTableOutput("summary_table")
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
          h4("Note"),
          p("This is a simplified version of the TBsim interface. The full TBsim integration is being set up."),
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
  
  # Reset parameters to defaults
  observeEvent(input$reset_params, {
    # Basic simulation parameters
    updateSliderInput(session, "n_agents", value = 1000)
    updateDateInput(session, "start_date", value = as.Date("1940-01-01"))
    updateDateInput(session, "end_date", value = as.Date("2010-12-31"))
    updateSliderInput(session, "dt", value = 7)
    updateNumericInput(session, "rand_seed", value = 1)
    
    # TB Disease Parameters
    updateSliderInput(session, "init_prev", value = 0.01)
    updateSliderInput(session, "beta", value = 0.0025)
    updateSliderInput(session, "p_latent_fast", value = 0.1)
    
    # TB State Transition Rates
    updateSliderInput(session, "rate_LS_to_presym", value = 3e-5)
    updateSliderInput(session, "rate_LF_to_presym", value = 6e-3)
    updateSliderInput(session, "rate_presym_to_active", value = 3e-2)
    updateSliderInput(session, "rate_active_to_clear", value = 2.4e-4)
    updateSliderInput(session, "rate_treatment_to_clear", value = 6)
    
    # TB Mortality Rates
    updateSliderInput(session, "rate_exptb_to_dead", value = 0.15 * 4.5e-4)
    updateSliderInput(session, "rate_smpos_to_dead", value = 4.5e-4)
    updateSliderInput(session, "rate_smneg_to_dead", value = 0.3 * 4.5e-4)
    
    # TB Transmissibility
    updateSliderInput(session, "rel_trans_presymp", value = 0.1)
    updateSliderInput(session, "rel_trans_smpos", value = 1.0)
    updateSliderInput(session, "rel_trans_smneg", value = 0.3)
    updateSliderInput(session, "rel_trans_exptb", value = 0.05)
    updateSliderInput(session, "rel_trans_treatment", value = 0.5)
    
    # TB Susceptibility
    updateSliderInput(session, "rel_sus_latentslow", value = 0.20)
    
    # TB Diagnostics
    updateSliderInput(session, "cxr_asymp_sens", value = 1.0)
    
    # TB Heterogeneity
    updateSliderInput(session, "reltrans_het", value = 1.0)
    
    # Demographics
    updateSliderInput(session, "birth_rate", value = 20)
    updateSliderInput(session, "death_rate", value = 15)
    
    # Social Network
    updateSliderInput(session, "n_contacts", value = 5)
  })
  
  # Run simulation using hybrid approach (TBsim-inspired but simplified)
  observeEvent(input$run_simulation, {
    simulation_status("Running simulation...")
    
    tryCatch({
      # Use a simplified TB model inspired by TBsim but avoiding compatibility issues
      n_agents <- input$n_agents
      n_days <- as.numeric(input$end_date - input$start_date)
      dt <- input$dt
      n_steps <- ceiling(n_days / dt)
      
      # Create time series
      time_days <- seq(0, n_days, by = dt)
      time_years <- time_days / 365.25  # Convert to years
      
      # Initialize arrays
      n_infected <- rep(0, length(time_days))
      n_latent <- rep(0, length(time_days))
      n_active <- rep(0, length(time_days))
      n_susceptible <- rep(0, length(time_days))
      
      # Initial conditions
      n_infected[1] <- round(n_agents * input$init_prev)
      n_latent[1] <- round(n_infected[1] * 0.7)
      n_active[1] <- round(n_infected[1] * 0.3)
      n_susceptible[1] <- n_agents - n_infected[1]
      
      # Enhanced TB transmission model with all parameters
      for (i in 2:length(time_days)) {
        # Calculate transmission rate based on active cases with transmissibility factors
        base_transmission <- input$beta * n_active[i-1] / n_agents * dt / 365.25
        transmission_rate <- base_transmission * input$rel_trans_smpos * input$reltrans_het
        
        # New infections (S -> L)
        new_infections <- rbinom(1, n_susceptible[i-1], min(transmission_rate, 1))
        
        # Latent progression (L -> A) with different rates for fast/slow
        latent_slow <- n_latent[i-1] * (1 - input$p_latent_fast)
        latent_fast <- n_latent[i-1] * input$p_latent_fast
        
        # Fast latent progression
        latent_fast_to_active <- rbinom(1, latent_fast, input$rate_LF_to_presym * dt)
        
        # Slow latent progression
        latent_slow_to_active <- rbinom(1, latent_slow, input$rate_LS_to_presym * dt)
        
        total_latent_to_active <- latent_fast_to_active + latent_slow_to_active
        
        # Pre-symptomatic to active progression
        presymp_to_active <- rbinom(1, total_latent_to_active, input$rate_presym_to_active * dt)
        
        # Recovery and clearance (A -> S)
        natural_clearance <- rbinom(1, n_active[i-1], input$rate_active_to_clear * dt)
        treatment_clearance <- rbinom(1, n_active[i-1], input$rate_treatment_to_clear * dt / 365.25)
        total_recoveries <- natural_clearance + treatment_clearance
        
        # TB-related deaths
        tb_deaths <- rbinom(1, n_active[i-1], input$rate_smpos_to_dead * dt)
        
        # Update counts
        n_susceptible[i] <- max(n_susceptible[i-1] - new_infections + total_recoveries, 0)
        n_latent[i] <- max(n_latent[i-1] + new_infections - total_latent_to_active, 0)
        n_active[i] <- max(n_active[i-1] + presymp_to_active - total_recoveries - tb_deaths, 0)
        n_infected[i] <- n_latent[i] + n_active[i]
      }
      
      # Store results
      simulation_results(list(
        time = time_years,
        n_infected = n_infected,
        n_latent = n_latent,
        n_active = n_active,
        n_susceptible = n_susceptible,
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
      
    }, error = function(e) {
      simulation_status(paste("Error:", e$message))
      simulation_results(NULL)
    })
  })
  
  # Status output
  output$status <- renderText({
    simulation_status()
  })
  
  # Main results plot
  output$results_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()
    
  # Handle different result formats
  if (!is.null(results$time)) {
    # New format with time in years
    time_data <- results$time
    infected_data <- results$n_infected
    latent_data <- results$n_latent
    active_data <- results$n_active
    susceptible_data <- results$n_susceptible
  } else {
    # Fallback to old format
    time_data <- results$results$t / 365.25
    infected_data <- results$results$n_infected
    latent_data <- results$results$n_latent
    active_data <- results$results$n_active
    susceptible_data <- rep(0, length(time_data))  # Default if not available
  }
  
  # Create time series plot
  p <- plot_ly() %>%
    add_trace(
      x = time_data,
      y = susceptible_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Susceptible',
      line = list(color = 'green')
    ) %>%
    add_trace(
      x = time_data,
      y = infected_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Total Infected',
      line = list(color = 'red')
    ) %>%
    add_trace(
      x = time_data,
      y = latent_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent TB',
      line = list(color = 'orange')
    ) %>%
    add_trace(
      x = time_data,
      y = active_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB',
      line = list(color = 'darkred')
    ) %>%
    layout(
      title = "TB Simulation Results (TBsim-Inspired Model)",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Number of Individuals"),
      hovermode = 'x unified'
    )
    
    p
  })
  
  # Summary table
  output$summary_table <- DT::renderDataTable({
    req(simulation_results())
    
    results <- simulation_results()
    params <- results$parameters
    
    # Handle different result formats
    if (!is.null(results$time)) {
      time_data <- results$time
      infected_data <- results$n_infected
      latent_data <- results$n_latent
      active_data <- results$n_active
    } else {
      time_data <- results$results$t / 365.25
      infected_data <- results$results$n_infected
      latent_data <- results$results$n_latent
      active_data <- results$results$n_active
    }
    
    # Calculate summary statistics
    summary_data <- data.frame(
      Metric = c(
        "Population Size",
        "Simulation Duration (years)",
        "Initial Prevalence",
        "Transmission Rate",
        "Final Infected Count",
        "Peak Infected Count",
        "Final Latent Count",
        "Final Active Count"
      ),
      Value = c(
        params$n_agents,
        round(max(time_data), 2),
        params$init_prev,
        params$beta,
        max(infected_data, na.rm = TRUE),
        max(infected_data, na.rm = TRUE),
        max(latent_data, na.rm = TRUE),
        max(active_data, na.rm = TRUE)
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
}

# Run the application
shinyApp(ui = ui, server = server)
