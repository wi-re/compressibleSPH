

def integrateRigidBody(rigidBody, dudt, dwdt, dt):
    rigidBody.angularVelocity += dwdt * dt
    rigidBody.linearVelocity += dudt * dt
    
    rigidBody.orientation += rigidBody.angularVelocity * dt
    rigidBody.centerOfMass += rigidBody.linearVelocity * dt

    return rigidBody
