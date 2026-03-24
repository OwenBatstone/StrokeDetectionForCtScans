//File For Login Page.
import 'package:flutter/material.dart';
import '../pages/stroke_zip_home.dart';
import '../login/sign_in.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  //text controllers
  //gets input from textboxes
  final GlobalKey<FormState> formKey = GlobalKey<FormState>();
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  bool _isLoading = false;

  //function to verify login State
  Future<void> _loginStates() async {
    if (!formKey.currentState!.validate()) return;
    //set state loading to avoid overlapping functions
    setState(() => _isLoading = true);
    try {
      //text from controllers
      final email = emailController.text.trim();
      final password = passwordController.text;

      //supabase function verifying email sign in
      await signInWithEmail(email, password);
      //if successfull push conext home and navigate home
      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => const StrokeZipHome()),
        );
      }
    } catch (e) {
      //error for loging in
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Login failed: ${e.toString()}')),
        );
      }
    } finally {
      //set is loading false now indicating not currently trying to login
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override //dispose of controllers
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
    super.dispose();
  }

  //Page layout
  @override
  Widget build(BuildContext context) {
    //media query to make elements on page relative size of device resolution
    final size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        //const home bar for title
        title: const Text("Stroke Zip Classifier + Locator"),
      ),
      body: Padding(
        // spacing og elements
        padding: EdgeInsets.only(
          //control height from top
          top: size.height * .1, //from top 10% offset from elements
        ),
        child: Align(
          //align top center
          alignment: Alignment.topCenter,
          child: SizedBox(
            //sized box for login card
            width: size.width * .8, //relative size to width of screen
            child: Card(
              elevation: 4, //dropshadow
              child: Padding(
                //padding edges withing login card
                padding: EdgeInsets.all(24),
                child: Form(
                  key: formKey,
                  child: Column(
                    //column aligning elements in card
                    mainAxisSize: MainAxisSize.min,
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      //Text box for Email entry
                      TextFormField(
                        controller:
                            emailController, //global text controller for getting email input
                        decoration: const InputDecoration(
                          labelText: "Email", //text shown
                          border: OutlineInputBorder(),
                        ),
                        //Validator to check for something
                        validator: (value) {
                          //this is where more sophisticated checking could be added "contatins @ ect"
                          if (value == null || value.isEmpty) {
                            return "Enter Email";
                          }
                          return null;
                        },
                      ),
                      const SizedBox(
                        //text feild for Password box
                        height: 20, //size of box height
                      ),
                      TextFormField(
                        controller:
                            passwordController, //controller to get password is value entered in text field
                        obscureText: true,
                        decoration: const InputDecoration(
                          labelText: "Password",
                          border: OutlineInputBorder(),
                        ),
                        validator: (value) {
                          //checking input
                          if (value == null || value.isEmpty) {
                            return "Enter Password";
                          }
                          return null;
                        },
                      ),
                      //Login Button
                      const SizedBox(height: 20),
                      //elevated button for submit information
                      ElevatedButton(
                        onPressed: _isLoading
                            ? null
                            : _loginStates, //if not currently tring to sign in call sign in logic
                        child:
                            _isLoading //if is loading show icon on button
                            ? const SizedBox(
                                height: 20,
                                width: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : const Text('Login'),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
