# Simple TBsim Shiny App - Basic Version
# This version focuses on getting the Shiny interface working first

library(shiny)
library(plotly)
library(DT)

# Define UI
ui <- fluidPage(
  titlePanel("TBsim - Tuberculosis Simulation Web Interface"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Simulation Parameters"),
      
      # Basic simulation parameters
      h4("Basic Settings"),
      sliderInput("n_agents", "Population Size", value = 1000, min = 100, max = 10000, step = 100),
      dateInput("start_date", "Start Date", value = "1940-01-01"),
      dateInput("end_date", "End Date", value = "2010-12-31"),
      sliderInput("dt", "Time Step (days)", value = 7, min = 1, max = 30, step = 1),
      sliderInput("rand_seed", "Random Seed", value = 1, min = 1, max = 10000, step = 1),
      
      # TB-specific parameters
      h4("TB Disease Parameters"),
      sliderInput("init_prev", "Initial Prevalence", value = 0.01, min = 0, max = 1, step = 0.001),
      sliderInput("beta", "Transmission Rate (per year)", value = 0.0025, min = 0, max = 0.1, step = 0.0001),
      sliderInput("p_latent_fast", "Probability of Fast Latent TB", value = 0.1, min = 0, max = 1, step = 0.01),
      
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
    updateSliderInput(session, "n_agents", value = 1000)
    updateDateInput(session, "start_date", value = as.Date("1940-01-01"))
    updateDateInput(session, "end_date", value = as.Date("2010-12-31"))
    updateSliderInput(session, "dt", value = 7)
    updateSliderInput(session, "rand_seed", value = 1)
    updateSliderInput(session, "init_prev", value = 0.01)
    updateSliderInput(session, "beta", value = 0.0025)
    updateSliderInput(session, "p_latent_fast", value = 0.1)
    updateSliderInput(session, "birth_rate", value = 20)
    updateSliderInput(session, "death_rate", value = 15)
    updateSliderInput(session, "n_contacts", value = 5)
  })
  
  # Run simulation (simplified version)
  observeEvent(input$run_simulation, {
    simulation_status("Running simulation...")
    
    tryCatch({
      # Simulate a simple TB transmission model
      n_agents <- input$n_agents
      n_days <- as.numeric(input$end_date - input$start_date)
      dt <- input$dt
      n_steps <- ceiling(n_days / dt)
      
      # Simple SIR-like model
      time_days <- seq(0, n_days, by = dt)
      time_years <- time_days / 365.25  # Convert to years
      n_infected <- rep(0, length(time_days))
      n_latent <- rep(0, length(time_days))
      n_active <- rep(0, length(time_days))
      
      # Initial conditions
      n_infected[1] <- round(n_agents * input$init_prev)
      n_latent[1] <- round(n_infected[1] * 0.7)
      n_active[1] <- round(n_infected[1] * 0.3)
      
      # Simple transmission model
      for (i in 2:length(time_days)) {
        # New infections
        new_infected <- round(n_infected[i-1] * input$beta * dt / 365.25)
        n_infected[i] <- min(n_infected[i-1] + new_infected, n_agents)
        
        # Latent to active progression
        latent_to_active <- round(n_latent[i-1] * input$p_latent_fast * dt / 365.25)
        n_latent[i] <- max(n_latent[i-1] - latent_to_active, 0)
        n_active[i] <- min(n_active[i-1] + latent_to_active, n_agents)
      }
      
      # Store results
      simulation_results(list(
        time = time_years,
        n_infected = n_infected,
        n_latent = n_latent,
        n_active = n_active,
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
    
    # Create time series plot
    p <- plot_ly() %>%
      add_trace(
        x = results$time,
        y = results$n_infected,
        type = 'scatter',
        mode = 'lines',
        name = 'Total Infected',
        line = list(color = 'red')
      ) %>%
      add_trace(
        x = results$time,
        y = results$n_latent,
        type = 'scatter',
        mode = 'lines',
        name = 'Latent TB',
        line = list(color = 'orange')
      ) %>%
      add_trace(
        x = results$time,
        y = results$n_active,
        type = 'scatter',
        mode = 'lines',
        name = 'Active TB',
        line = list(color = 'darkred')
      ) %>%
      layout(
        title = "TB Simulation Results",
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
        round(max(results$time), 2),
        params$init_prev,
        params$beta,
        max(results$n_infected, na.rm = TRUE),
        max(results$n_infected, na.rm = TRUE),
        max(results$n_latent, na.rm = TRUE),
        max(results$n_active, na.rm = TRUE)
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
