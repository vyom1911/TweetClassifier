
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Tweet Classifier</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
  
		<script>
			$( document ).ready(function() {
				$( "#butTweet" ).click(function() {
					$('#result').html('Pending ...');
					var tweetStr = $('#tweet').val()
					
					$.ajax({
						type: 'POST',
						url: "script/classifier_old.py",
						data: { param: tweetStr}, 
						success: function (response) {													
							
							$('#result').html( response );
						},
						error: function () {
							alert("Not Working");
						}
					});
				});
			});
				
			
		</script>
  <style>
    /* Remove the navbar's default margin-bottom and rounded borders */ 
    .navbar {
      margin-bottom: 0;
      border-radius: 0;
	  
    }
    
    /* Set height of the grid so .sidenav can be 100% (adjust as needed) */
    .row.content {height: 450px}
    
    /* Set gray background color and 100% height */
    .sidenav {
      padding-top: 20px;
      background-color: #E2E2E2;
      height: 100%;
    }
    
    /* Set black background color, white text and some padding */
    footer {
      background-color: #4E52FA; /*#555*/
      color: white;
      padding: 15px;
    }
    
    /* On small screens, set height to 'auto' for sidenav and grid */
    @media screen and (max-width: 767px) {
      .sidenav {
        height: auto;
        padding: 15px;
      }
      .row.content {height:auto;} 
    }
  </style>
</head>
<body>

<nav class="navbar navbar-inverse">
  <div class="container-fluid">
    <div class="navbar-header">
      <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#myNavbar">
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>                        
      </button>
      <a class="navbar-brand" href=""><img src="images/logo.png" /></a>
    </div>
    <div class="collapse navbar-collapse" id="myNavbar">
      <ul class="nav navbar-nav">
        <li class="active"><a href="http://localhost:8080/classifyTweets/">Home</a></li>
        <li><a href="http://localhost:8080/classifyTweets/svm.php">SVM</a></li>
        <li><a href="#">Projects</a></li>
        <li><a href="#">Contact</a></li>
      </ul>
    </div>
  </div>
</nav>
  
<div class="container-fluid text-center">    
  <div class="row content">
    <div class="col-sm-2 sidenav">
		
    </div>
    <div class="col-sm-8 text-left"> 
		<h1>Welcome (using Naive Bayes)</h1>
		<p>You can find out whether a tweet is for emergency in crisis or not</p>
		<hr>
		<form>
			<div class="form-group">
			  <label for="tweet">Please type your tweet here:</label>
			  <input type="text" class="form-control" id="tweet">
			</div>
			<button id="butTweet" type="button" class="btn btn-primary">Check</button>
		</form>
    </div>
	<div class="col-sm-8 text-left"> 
		<h2>Result</h2>
		<div id="result" class="well"></div>
    </div>
  </div>
</div>

<footer class="container-fluid text-center">
  <p>Adavance Information Systems Course - Fall 2017</p>
</footer>

</body>
</html>
