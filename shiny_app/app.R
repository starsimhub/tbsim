# TBsim Shiny Web Application
# A web interface for running tuberculosis simulations using the tbsim package

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
      numericInput("n_agents", "Population Size", value = 1000, min = 100, max = 10000, step = 100),
      dateInput("start_date", "Start Date", value = "1940-01-01"),
      dateInput("end_date", "End Date", value = "2010-12-31"),
      numericInput("dt", "Time Step (days)", value = 7, min = 1, max = 30),
      numericInput("rand_seed", "Random Seed", value = 1, min = 1, max = 10000),
      
      # TB-specific parameters
      h4("TB Disease Parameters"),
      numericInput("init_prev", "Initial Prevalence", value = 0.01, min = 0, max = 1, step = 0.001),
      numericInput("beta", "Transmission Rate (per year)", value = 0.0025, min = 0, max = 0.1, step = 0.0001),
      numericInput("p_latent_fast", "Probability of Fast Latent TB", value = 0.1, min = 0, max = 1, step = 0.01),
      
      # Demographics
      h4("Demographics"),
      numericInput("birth_rate", "Birth Rate (per 1000)", value = 20, min = 0, max = 100),
      numericInput("death_rate", "Death Rate (per 1000)", value = 15, min = 0, max = 100),
      
      # Network parameters
      h4("Social Network"),
      numericInput("n_contacts", "Average Contacts per Person", value = 5, min = 1, max = 50),
      
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
        
        # Plots tab
        tabPanel("Detailed Plots",
          h3("Detailed Analysis"),
          plotlyOutput("detailed_plot", height = "600px"),
          br(),
          h4("State Transitions"),
          plotlyOutput("transitions_plot", height = "400px")
        ),
        
        # Data tab
        tabPanel("Raw Data",
          h3("Raw Simulation Data"),
          DT::dataTableOutput("raw_data_table")
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
  
  # Reset parameters to defaults
  observeEvent(input$reset_params, {
    updateNumericInput(session, "n_agents", value = 1000)
    updateDateInput(session, "start_date", value = as.Date("1940-01-01"))
    updateDateInput(session, "end_date", value = as.Date("2010-12-31"))
    updateNumericInput(session, "dt", value = 7)
    updateNumericInput(session, "rand_seed", value = 1)
    updateNumericInput(session, "init_prev", value = 0.01)
    updateNumericInput(session, "beta", value = 0.0025)
    updateNumericInput(session, "p_latent_fast", value = 0.1)
    updateNumericInput(session, "birth_rate", value = 20)
    updateNumericInput(session, "death_rate", value = 15)
    updateNumericInput(session, "n_contacts", value = 5)
  })
  
  # Run simulation
  observeEvent(input$run_simulation, {
    simulation_status("Running simulation...")
    
    tryCatch({
      # Build simulation parameters
      spars <- list(
        unit = 'day',
        dt = input$dt,
        start = sciris$date(input$start_date),
        stop = sciris$date(input$end_date),
        rand_seed = input$rand_seed,
        verbose = 0
      )
      
      # Create population
      pop <- starsim$People(n_agents = input$n_agents)
      
      # Create TB disease model
      tb_pars <- list(
        unit = 'day',
        dt = input$dt,
        beta = starsim$rate_prob(input$beta, unit = 'year'),
        init_prev = starsim$bernoulli(p = input$init_prev),
        p_latent_fast = starsim$bernoulli(p = input$p_latent_fast)
      )
      tb <- tbsim$TB(tb_pars)
      
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
        pars = spars
      )
      
      # Run simulation
      sim$run()
      
      # Extract results
      results <- sim$results$flatten()
      
      # Store results
      simulation_results(list(
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
    
    results <- simulation_results()$results
    
    # Create time series plot
    p <- plot_ly() %>%
      add_trace(
        x = results$t,
        y = results$n_infected,
        type = 'scatter',
        mode = 'lines',
        name = 'Infected',
        line = list(color = 'red')
      ) %>%
      add_trace(
        x = results$t,
        y = results$n_latent,
        type = 'scatter',
        mode = 'lines',
        name = 'Latent',
        line = list(color = 'orange')
      ) %>%
      add_trace(
        x = results$t,
        y = results$n_active,
        type = 'scatter',
        mode = 'lines',
        name = 'Active',
        line = list(color = 'darkred')
      ) %>%
      layout(
        title = "TB Simulation Results",
        xaxis = list(title = "Time"),
        yaxis = list(title = "Number of Individuals"),
        hovermode = 'x unified'
      )
    
    p
  })
  
  # Detailed plot
  output$detailed_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()$results
    
    # Create subplot with multiple metrics
    p1 <- plot_ly(
      x = results$t,
      y = results$n_infected,
      type = 'scatter',
      mode = 'lines',
      name = 'Total Infected'
    ) %>%
      layout(yaxis = list(title = "Count"))
    
    p2 <- plot_ly(
      x = results$t,
      y = results$n_latent,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent TB'
    ) %>%
      layout(yaxis = list(title = "Count"))
    
    p3 <- plot_ly(
      x = results$t,
      y = results$n_active,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB'
    ) %>%
      layout(yaxis = list(title = "Count"))
    
    subplot(p1, p2, p3, nrows = 3, shareX = TRUE) %>%
      layout(
        title = "Detailed TB Simulation Metrics",
        showlegend = TRUE
      )
  })
  
  # Transitions plot
  output$transitions_plot <- renderPlotly({
    req(simulation_results())
    
    results <- simulation_results()$results
    
    # Calculate transition rates if available
    if ("n_new_infections" %in% names(results)) {
      p <- plot_ly(
        x = results$t,
        y = results$n_new_infections,
        type = 'scatter',
        mode = 'lines',
        name = 'New Infections'
      ) %>%
        layout(
          title = "TB State Transitions",
          xaxis = list(title = "Time"),
          yaxis = list(title = "New Cases per Time Step")
        )
    } else {
      # Fallback plot
      p <- plot_ly(
        x = results$t,
        y = results$n_infected,
        type = 'scatter',
        mode = 'lines',
        name = 'Total Infected'
      ) %>%
        layout(
          title = "TB State Transitions",
          xaxis = list(title = "Time"),
          yaxis = list(title = "Number of Individuals")
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
        "Final Latent Count",
        "Final Active Count"
      ),
      Value = c(
        params$n_agents,
        round((params$end_date - params$start_date) / 365.25, 2),
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
  
  # Raw data table
  output$raw_data_table <- DT::renderDataTable({
    req(simulation_results())
    
    results <- simulation_results()$results
    
    # Convert results to data frame
    df <- data.frame(
      Time = results$t,
      Infected = results$n_infected,
      Latent = results$n_latent,
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
}

# Run the application
shinyApp(ui = ui, server = server)
