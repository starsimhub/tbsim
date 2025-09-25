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
  # Custom header with logo
  div(
    style = "display: flex; align-items: center; margin-bottom: 20px; 
             padding: 10px; background-color: #f8f9fa; border-radius: 5px;",
    img(src = "logo.png", height = 60, style = "margin-right: 15px; vertical-align: middle;"),
    h2("TBsim - Tuberculosis Simulation Web Interface", style = "margin: 0; color: #333;")
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
          plotlyOutput("results_plot", height = "600px"),
          br(),
          h4("Summary Statistics"),
          DT::dataTableOutput("summary_table"),
          br(),
          h4("Simulation Parameters Used"),
          DT::dataTableOutput("parameters_table"),
          br(),
          h4("Raw Simulation Parameters (my_pars)"),
          verbatimTextOutput("my_pars_output")
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
  simulation_running <- reactiveVal(FALSE)
  my_pars <- reactiveVal(NULL)
  
  # Reset parameters to defaults
  observeEvent(input$reset_params, {
    updateSliderInput(session, "n_agents", value = 1000)
    updateDateInput(session, "start_date", value = as.Date("1940-01-01"))
    updateDateInput(session, "end_date", value = as.Date("2010-12-31"))
    updateSliderInput(session, "dt", value = 7)
    updateNumericInput(session, "rand_seed", value = 1)
    updateSliderInput(session, "init_prev", value = 0.01)
    updateSliderInput(session, "beta", value = 0.0025)
    updateSliderInput(session, "p_latent_fast", value = 0.1)
    updateSliderInput(session, "birth_rate", value = 20)
    updateSliderInput(session, "death_rate", value = 15)
    updateSliderInput(session, "n_contacts", value = 5)
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
        n_latent = as.numeric(results$tb_n_latent_slow$tolist()) + as.numeric(results$tb_n_latent_fast$tolist()),
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
      latent_data <- results$n_latent
      active_data <- results$n_active
      susceptible_data <- results$n_susceptible
      presymp_data <- results$n_presymp
    } else {
      # Fallback format
      time_data <- results$time
      infected_data <- results$n_infected
      latent_data <- results$n_latent
      active_data <- results$n_active
      susceptible_data <- results$n_susceptible
      presymp_data <- results$n_presymp
    }
    
    # Create time series plot
    p <- plot_ly() %>%
    add_trace(
      x = time_data,
      y = susceptible_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Susceptible',
      line = list(color = '#440154')
    ) %>%
    add_trace(
      x = time_data,
      y = infected_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Total Infected',
      line = list(color = '#31688e')
    ) %>%
    add_trace(
      x = time_data,
      y = latent_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent TB',
      line = list(color = '#35b779')
    ) %>%
    add_trace(
      x = time_data,
      y = presymp_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Pre-symptomatic',
      line = list(color = '#fde725')
    ) %>%
    add_trace(
      x = time_data,
      y = active_data,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB',
      line = list(color = '#e16462')
    ) %>%
    layout(
      title = "TB Simulation Results (Real TBsim Model)",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Number of Individuals", type = "log"),
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
      layout(yaxis = list(title = "Count", type = "log"))
    
    p2 <- plot_ly(
      x = results$t,
      y = results$n_latent,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent TB'
    ) %>%
      layout(yaxis = list(title = "Count", type = "log"))
    
    p3 <- plot_ly(
      x = results$t,
      y = results$n_active,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB'
    ) %>%
      layout(yaxis = list(title = "Count", type = "log"))
    
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
          yaxis = list(title = "New Cases per Time Step", type = "log")
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
        "Final Latent Count",
        "Final Active Count"
      ),
      Value = c(
        params$n_agents,
        round((params$end_date - params$start_date) / 365.25, 2),
        params$init_prev,
        params$beta,
        max(as.numeric(results$tb_n_infected$tolist()), na.rm = TRUE),
        max(as.numeric(results$tb_n_infected$tolist()), na.rm = TRUE),
        max(as.numeric(results$tb_n_latent_slow$tolist()) + 
            as.numeric(results$tb_n_latent_fast$tolist()), na.rm = TRUE),
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
        "TB Diagnostics",
        "TB Heterogeneity",
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
        "CXR Asymptomatic Sensitivity",
        "Transmission Heterogeneity",
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
        input$cxr_asymp_sens,
        input$reltrans_het,
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
    req(my_pars())
    
    tryCatch({
      # Convert SimPars object to dictionary first, then to R list
      pars_dict <- my_pars()$to_dict()
      pars_list <- py_to_r(pars_dict)
      
      # Create a detailed string representation
      output_lines <- c()
      output_lines <- c(output_lines, "=== SIMULATION PARAMETERS (my_pars) ===")
      output_lines <- c(output_lines, paste("Type:", class(pars_list)))
      output_lines <- c(output_lines, paste("Length:", length(pars_list)))
      output_lines <- c(output_lines, "")
      
      # Add each parameter with its value
      for (i in seq_along(pars_list)) {
        param_name <- names(pars_list)[i]
        param_value <- pars_list[[i]]
        
        output_lines <- c(output_lines, paste("Parameter", i, ":", param_name))
        output_lines <- c(output_lines, paste("  Type:", class(param_value)))
        output_lines <- c(output_lines, paste("  Value:", toString(param_value)))
        output_lines <- c(output_lines, "")
      }
      
      paste(output_lines, collapse = "\n")
    }, error = function(e) {
      # Fallback: try to get basic info about the object
      paste("Error converting my_pars:", e$message, "\n",
            "Object type:", class(my_pars()), "\n",
            "Object length:", length(my_pars()), "\n",
            "Note: sim.pars is a SimPars object, not a regular dictionary")
    })
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
