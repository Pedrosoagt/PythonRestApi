"""This module will serve the api request."""

from config import client
from app import app
from bson.json_util import dumps
from flask import request, jsonify
import json
import ast
import imp

# Import the helpers module
helper_module = imp.load_source('*', './app/helpers.py')

# Select the database
db = client.development
# Select the collection
collection = db.pads

@app.route("/pads", methods=['POST'])
def create_pad():
    """
       Function to create new pads.
       """
    try:
        # Create new pads
        try:
            body = ast.literal_eval(json.dumps(request.get_json()))
        except:
            # Bad request as request body is not available
            # Add message for debugging purpose
            return "", 400

        record_created = collection.insert(body)

        # Prepare the response
        if isinstance(record_created, list):
            # Return list of Id of the newly created item
            return jsonify([str(v) for v in record_created]), 201
        else:
            # Return Id of the newly created item
            return jsonify(str(record_created)), 201
    except:
        # Error while trying to create the resource
        # Add message for debugging purpose
        return "", 500
        
""""
TODO
@app.route("/pads/<xml_path>", methods=['GET'])
"""

@app.route("/pads", methods=['GET'])
def fetch_pads():
    """
       Function to fetch the pads.
    """
    try:
        # Call the function to get the query params
        query_params = helper_module.parse_query_params(request.query_string)
        # Check if dictionary is not empty
        if query_params:

            # Try to convert the value to int
            query = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in query_params.items()}

            # Fetch all the record(s)
            records_fetched = collection.find(query)

            # Check if the records are found
            if records_fetched.count() > 0:
                # Prepare the response
                return dumps(records_fetched)
            else:
                # No records are found
                return "", 404

        # If dictionary is empty
        else:
            # Return all the records as query string parameters are not available
            if collection.find().count > 0:
                # Prepare response if the pads are found
                return dumps(collection.find())
            else:
                # Return empty array if no pads are found
                return jsonify([])
    except:
        # Error while trying to fetch the resource
        # Add message for debugging purpose
        return "", 500

@app.route("/pads/<pad_id>", methods=['POST'])
def update_pad(pad_id):
    """
       Function to update the pad.
       """
    try:
        # Get the value which needs to be updated
        try:
            body = ast.literal_eval(json.dumps(request.get_json()))
        except:
            # Bad request as the request body is not available
            # Add message for debugging purpose
            return "", 400

        # Updating the pad
        records_updated = collection.update_one({"id": int(pad_id)}, body)

        # Check if resource is updated
        if records_updated.modified_count > 0:
            # Prepare the response as resource is updated successfully
            return "", 200
        else:
            # Bad request as the resource is not available to update
            # Add message for debugging purpose
            return "", 404
    except:
        # Error while trying to update the resource
        # Add message for debugging purpose
        return "", 500

@app.route("/pads/<pad_id>", methods=['DELETE'])
def remove_pad(pad_id):
    """
       Function to remove the pad.
    """
    try:
        # Delete the pad
        delete_pad = collection.delete_one({"id": int(pad_id)})

        if delete_pad.deleted_count > 0 :
            # Prepare the response
            return "", 204
        else:
            # Resource Not found
            return "", 404
    except:
        # Error while trying to delete the resource
        # Add message for debugging purpose
        return "", 500

@app.errorhandler(404)
def page_not_found(e):
    """Send message to the pad with notFound 404 status."""
    # Message to the pad
    message = {
        "err":
            {
                "msg": "This route is currently not supported."
            }
    }
    # Making the message looks good
    resp = jsonify(message)
    # Sending OK response
    resp.status_code = 404
    # Returning the object
    return resp
